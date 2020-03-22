#  Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import sys

import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.initializer import Constant

from models.pointnet2_modules import MLP, pointnet_sa_module, conv_bn
from models.loss_utils import sigmoid_focal_loss , get_reg_loss
from utils.proposal_target import get_proposal_target_func
from utils.cyops.kitti_utils import rotate_pc_along_y

__all__ = ['RCNN']


class RCNN(object):
    def __init__(self, cfg, num_classes, batch_size, mode='TRAIN', use_xyz=True, input_channels=0):
        self.cfg = cfg
        self.use_xyz = use_xyz
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.inputs = None
        self.training = mode == 'TRAIN'
        self.batch_size = batch_size

    def create_tmp_var(self, name, dtype, shape):
        return fluid.default_main_program().current_block().create_var(
            name=name, dtype=dtype, shape=shape
        )

    def build_model(self, inputs):
        self.inputs = inputs
        if self.cfg.RCNN.ROI_SAMPLE_JIT:
            if self.training:
                proposal_target = get_proposal_target_func(self.cfg)
                        
                tmp_list = [
                    self.inputs['seg_mask'],
                    self.inputs['rpn_features'],
                    self.inputs['gt_boxes3d'],
                    self.inputs['rpn_xyz'],
                    self.inputs['pts_depth'],
                    self.inputs['roi_boxes3d'],
                    self.inputs['rpn_intensity'],
                ]
                out_name = ['reg_valid_mask' ,'sampled_pts' ,'roi_boxes3d', 'gt_of_rois', 'pts_feature' ,'cls_label','gt_iou']
                reg_valid_mask = self.create_tmp_var(name="reg_valid_mask",dtype='float32',shape=[-1,])
                sampled_pts = self.create_tmp_var(name="sampled_pts",dtype='float32',shape=[-1, self.cfg.RCNN.NUM_POINTS, 3])
                new_roi_boxes3d = self.create_tmp_var(name="new_roi_boxes3d",dtype='float32',shape=[-1, 7])
                gt_of_rois = self.create_tmp_var(name="gt_of_rois", dtype='float32', shape=[-1,7])
                pts_feature = self.create_tmp_var(name="pts_feature", dtype='float32',shape=[-1,512,130])
                cls_label = self.create_tmp_var(name="cls_label",dtype='int64',shape=[-1])
                gt_iou = self.create_tmp_var(name="gt_iou",dtype='float32',shape=[-1])
                
                out_list = [reg_valid_mask, sampled_pts, new_roi_boxes3d, gt_of_rois, pts_feature, cls_label, gt_iou]
                out = fluid.layers.py_func(func=proposal_target,x=tmp_list,out=out_list)
                
                self.target_dict = {}
                for i,item in enumerate(out):
                    self.target_dict[out_name[i]] = item
                
                pts = fluid.layers.concat(input=[self.target_dict['sampled_pts'],self.target_dict['pts_feature']], axis=2)
                self.debug = pts
                self.target_dict['pts_input'] = pts
            else:
                rpn_xyz, rpn_features = inputs['rpn_xyz'], inputs['rpn_features']
                batch_rois = inputs['roi_boxes3d']
                rpn_intensity = inputs['rpn_intensity']
                rpn_intensity = fluid.layers.unsqueeze(rpn_intensity,axes=[2])
                seg_mask = fluid.layers.unsqueeze(inputs['seg_mask'],axes=[2])
                if self.cfg.RCNN.USE_INTENSITY:
                    pts_extra_input_list = [rpn_intensity, seg_mask]
                else:
                    pts_extra_input_list = [seg_mask]

                if self.cfg.RCNN.USE_DEPTH:
                    pts_depth = inputs['pts_depth'] / 70.0 -0.5
                    pts_depth = fluid.layers.unsqueeze(pts_depth,axes=[2])
                    pts_extra_input_list.append(pts_depth)
                pts_extra_input = fluid.layers.concat(pts_extra_input_list, axis=2)
                pts_feature = fluid.layers.concat([pts_extra_input, rpn_features],axis=2)
                
                pooled_features, pooled_empty_flag = fluid.layers.roi_pool_3d(rpn_xyz,pts_feature,batch_rois,
                                                                              self.cfg.RCNN.POOL_EXTRA_WIDTH,
                                                                              sampled_pt_num=self.cfg.RCNN.NUM_POINTS)
                # canonical transformation
                batch_size = batch_rois.shape[0]
                roi_center = batch_rois[:, :, 0:3]
                tmp = pooled_features[:, :, :, 0:3] - fluid.layers.unsqueeze(roi_center,axes=[2])
                pooled_features = fluid.layers.concat(input=[tmp,pooled_features[:,:,:,3:]],axis=3)
                concat_list = []
                for i in range(batch_size):
                    tmp = rotate_pc_along_y(pooled_features[i, :, :, 0:3],
                                                        batch_rois[i, :, 6])
                    concat = fluid.layers.concat([tmp,pooled_features[i,:,:,3:]],axis=-1)
                    concat = fluid.layers.unsqueeze(concat,axes=[0])
                    concat_list.append(concat)
                pooled_features = fluid.layers.concat(concat_list,axis=0)
                pts = fluid.layers.reshape(pooled_features,shape=[-1,pooled_features.shape[2],pooled_features.shape[3]])
        
        else:
            pts = inputs['pts_input']
            self.target_dict = {}
            self.target_dict['pts_input'] = inputs['pts_input']
            self.target_dict['roi_boxes3d'] = inputs['roi_boxes3d']
        
            if self.training:
                self.target_dict['cls_label'] = inputs['cls_label']
                self.target_dict['reg_valid_mask'] = inputs['reg_valid_mask']
                self.target_dict['gt_of_rois'] = inputs['gt_boxes3d_ct']
        
        xyz = pts[:,:,0:3]
        feature = fluid.layers.transpose(pts[:,:,3:], [0,2,1]) if pts.shape[-1]>3 else None
        if self.cfg.RCNN.USE_RPN_FEATURES:
            self.rcnn_input_channel = 3 + int(self.cfg.RCNN.USE_INTENSITY) + \
                                      int(self.cfg.RCNN.USE_MASK) + int(self.cfg.RCNN.USE_DEPTH)
            c_out = self.cfg.RCNN.XYZ_UP_LAYER[-1]

            xyz_input = pts[:,:,:self.rcnn_input_channel]
            xyz_input = fluid.layers.transpose(xyz_input, [0,2,1])
            xyz_input = fluid.layers.unsqueeze(xyz_input, axes=[3])
            
            rpn_feature = pts[:,:,self.rcnn_input_channel:]
            rpn_feature = fluid.layers.transpose(rpn_feature, [0,2,1])
            rpn_feature = fluid.layers.unsqueeze(rpn_feature,axes=[3])

            xyz_feature = MLP(
                xyz_input,
                out_channels_list=self.cfg.RCNN.XYZ_UP_LAYER,
                bn=self.cfg.RCNN.USE_BN,
                name="xyz_up_layer")
            
            merged_feature = fluid.layers.concat([xyz_feature, rpn_feature],axis=1)
            merged_feature = MLP(
                merged_feature,
                out_channels_list=[c_out], 
                bn=self.cfg.RCNN.USE_BN, 
                name="xyz_down_layer")

            xyzs = [xyz]
            features = [fluid.layers.squeeze(merged_feature,axes=[3])]
        else:
            xyzs = [xyz]
            features = [feature]
        
        # forward
        xyzi, featurei = xyzs[-1], features[-1]
        for k in range(len(self.cfg.RCNN.SA_CONFIG.NPOINTS)):
            mlps = self.cfg.RCNN.SA_CONFIG.MLPS[k]
            npoint = self.cfg.RCNN.SA_CONFIG.NPOINTS[k] if self.cfg.RCNN.SA_CONFIG.NPOINTS[k] != -1 else None
            
            xyzi, featurei = pointnet_sa_module(
                xyz=xyzi,
                feature = featurei,
                bn = self.cfg.RCNN.USE_BN,
                use_xyz = self.use_xyz,
                name = "sa_{}".format(k),
                npoint = npoint,
                mlps = [mlps],
                radiuss = [self.cfg.RCNN.SA_CONFIG.RADIUS[k]],
                nsamples = [self.cfg.RCNN.SA_CONFIG.NSAMPLE[k]]
            )
            xyzs.append(xyzi)
            features.append(featurei)
        
        head_in = features[-1]
        head_in = fluid.layers.unsqueeze(head_in, axes=[2])
        
        cls_out = head_in
        reg_out = cls_out
        
        for i in range(0, self.cfg.RCNN.CLS_FC.__len__()):
            cls_out = conv_bn(cls_out, self.cfg.RCNN.CLS_FC[i], bn=self.cfg.RCNN.USE_BN, name='rcnn_cls_{}'.format(i))
            if i == 0 and self.cfg.RCNN.DP_RATIO >= 0:
                cls_out = fluid.layers.dropout(cls_out, self.cfg.RCNN.DP_RATIO, dropout_implementation="upscale_in_train")
        cls_channel = 1 if self.num_classes == 2 else self.num_classes
        cls_out = conv_bn(cls_out, cls_channel, act=None, name="cls_out", bn=self.cfg.RCNN.USE_BN)
        self.cls_out = fluid.layers.squeeze(cls_out,axes=[1,3])
       
        per_loc_bin_num = int(self.cfg.RCNN.LOC_SCOPE / self.cfg.RCNN.LOC_BIN_SIZE) * 2
        loc_y_bin_num = int(self.cfg.RCNN.LOC_Y_SCOPE / self.cfg.RCNN.LOC_Y_BIN_SIZE) * 2
        reg_channel = per_loc_bin_num * 4 + self.cfg.RCNN.NUM_HEAD_BIN * 2 + 3
        reg_channel += (1 if not self.cfg.RCNN.LOC_Y_BY_BIN else loc_y_bin_num * 2)
        for i in range(0, self.cfg.RCNN.REG_FC.__len__()):
            reg_out = conv_bn(reg_out, self.cfg.RCNN.REG_FC[i], bn=self.cfg.RCNN.USE_BN, name='rcnn_reg_{}'.format(i))
            if i == 0 and self.cfg.RCNN.DP_RATIO >= 0:
                reg_out = fluid.layers.dropout(reg_out, self.cfg.RCNN.DP_RATIO, dropout_implementation="upscale_in_train")

        reg_out = conv_bn(reg_out, reg_channel, act=None, name="reg_out", bn=self.cfg.RCNN.USE_BN)
        self.reg_out = fluid.layers.squeeze(reg_out, axes=[2,3])

        
        self.outputs = {
            'rcnn_cls':self.cls_out,
            'rcnn_reg':self.reg_out,
        }
        if self.training:
            self.outputs.update(self.target_dict)
        elif not self.training:
            self.outputs['sample_id'] = inputs['sample_id']
            self.outputs['pts_input'] = inputs['pts_input']
            self.outputs['roi_boxes3d'] = inputs['roi_boxes3d']
            self.outputs['roi_scores'] = inputs['roi_scores']
            self.outputs['gt_iou'] = inputs['gt_iou'] 
            self.outputs['gt_boxes3d'] = inputs['gt_boxes3d']

            if self.cls_out.shape[1] == 1:
                raw_scores = fluid.layers.reshape(self.cls_out, shape=[-1])
                norm_scores = fluid.layers.sigmoid(raw_scores)
            else:
                norm_scores = fluid.layers.softmax(self.cls_out, axis=1)
            self.outputs['norm_scores'] = norm_scores
            
    def get_outputs(self):
        return self.outputs

    def get_loss(self):
        assert self.inputs is not None, \
            "please call build() first"
        rcnn_cls_label = self.outputs['cls_label']
        reg_valid_mask = self.outputs['reg_valid_mask']
        roi_boxes3d = self.outputs['roi_boxes3d']
        roi_size = roi_boxes3d[:, 3:6]
        gt_boxes3d_ct = self.outputs['gt_of_rois']
        pts_input = self.outputs['pts_input']

        rcnn_cls = self.cls_out
        rcnn_reg = self.reg_out

        # RCNN classification loss
        assert self.cfg.RCNN.LOSS_CLS in ["SigmoidFocalLoss", "BinaryCrossEntropy"], \
                "unsupported RCNN cls loss type {}".format(self.cfg.RCNN.LOSS_CLS)

        if self.cfg.RCNN.LOSS_CLS == "SigmoidFocalLoss":
            cls_flat = fluid.layers.reshape(self.cls_out, shape=[-1])
            cls_label_flat = fluid.layers.reshape(rcnn_cls_label, shape=[-1])
            cls_label_flat = fluid.layers.cast(cls_label_flat, dtype=cls_flat.dtype)
            cls_target = fluid.layers.cast(cls_label_flat>0, dtype=cls_flat.dtype)
            cls_label_flat.stop_gradient = True
            pos = fluid.layers.cast(cls_label_flat > 0, dtype=cls_flat.dtype)
            pos.stop_gradient = True
            pos_normalizer = fluid.layers.reduce_sum(pos)
            cls_weights = fluid.layers.cast(cls_label_flat >= 0, dtype=cls_flat.dtype)
            cls_weights = cls_weights / fluid.layers.clip(pos_normalizer, min=1.0, max=1e10)
            cls_weights.stop_gradient = True
            rcnn_loss_cls = sigmoid_focal_loss(cls_flat, cls_target, cls_weights)
            rcnn_loss_cls = fluid.layers.reduce_sum(rcnn_loss_cls)
        else: # BinaryCrossEntropy
            cls_label = fluid.layers.reshape(rcnn_cls_label, shape=self.cls_out.shape)
            cls_valid_mask = fluid.layers.cast(cls_label >= 0, dtype=self.cls_out.dtype)
            cls_label = fluid.layers.cast(cls_label, dtype=self.cls_out.dtype)
            cls_label.stop_gradient = True
            rcnn_loss_cls = fluid.layers.sigmoid_cross_entropy_with_logits(self.cls_out, cls_label)
            cls_mask_normalzer = fluid.layers.reduce_sum(cls_valid_mask)
            rcnn_loss_cls = fluid.layers.reduce_sum(rcnn_loss_cls * cls_valid_mask) \
                                / fluid.layers.clip(cls_mask_normalzer, min=1.0, max=1e10)

        # RCNN regression loss
        reg_out = self.reg_out
        fg_mask = fluid.layers.cast(reg_valid_mask > 0, dtype=reg_out.dtype)
        fg_mask = fluid.layers.unsqueeze(fg_mask, axes=[1])
        fg_mask.stop_gradient = True
        gt_boxes3d_ct = fluid.layers.reshape(gt_boxes3d_ct, [-1,7])
        all_anchor_size = roi_size
        anchor_size = all_anchor_size[fg_mask] if self.cfg.RCNN.SIZE_RES_ON_ROI else self.cfg.CLS_MEAN_SIZE[0]

        loc_loss, angle_loss, size_loss, loss_dict = get_reg_loss(
            reg_out * fg_mask, 
            gt_boxes3d_ct,
            fg_mask,
            point_num=float(self.batch_size*64),
            loc_scope=self.cfg.RCNN.LOC_SCOPE,
            loc_bin_size=self.cfg.RCNN.LOC_BIN_SIZE,
            num_head_bin=self.cfg.RCNN.NUM_HEAD_BIN,
            anchor_size=anchor_size,
            get_xz_fine=True,
            get_y_by_bin=self.cfg.RCNN.LOC_Y_BY_BIN,
            loc_y_scope=self.cfg.RCNN.LOC_Y_SCOPE,
            loc_y_bin_size=self.cfg.RCNN.LOC_Y_BIN_SIZE,
            get_ry_fine=True
        )
        rcnn_loss_reg = loc_loss + angle_loss + size_loss * 3
        rcnn_loss = rcnn_loss_cls + rcnn_loss_reg
        return rcnn_loss, rcnn_loss_cls, rcnn_loss_reg

