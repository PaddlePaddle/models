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

import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.initializer import Normal, Constant

from utils.proposal_utils import get_proposal_func
from models.pointnet2_msg import PointNet2MSG
from models.pointnet2_modules import conv_bn
from models.loss_utils import sigmoid_focal_loss, get_reg_loss


__all__ = ["RPN"]


class RPN(object):
    def __init__(self, cfg, batch_size, use_xyz=True, mode='TRAIN', prog=None):
        self.cfg = cfg
        self.batch_size = batch_size
        self.use_xyz = use_xyz
        self.mode = mode
        self.is_train = mode == 'TRAIN'
        self.inputs = None
        self.prog = fluid.default_main_program() if prog is None else prog

    def build(self, inputs):
        assert self.cfg.RPN.BACKBONE == 'pointnet2_msg', \
                "RPN backbone only support pointnet2_msg"
        self.inputs = inputs
        self.outputs = {}

        xyz = inputs["pts_input"]
        assert not self.cfg.RPN.USE_INTENSITY, \
                "RPN.USE_INTENSITY not support now"
        feature = None
        msg = PointNet2MSG(self.cfg, xyz, feature, self.use_xyz)
        backbone_xyz, backbone_feature = msg.build()
        self.outputs['backbone_xyz'] = backbone_xyz
        self.outputs['backbone_feature'] = backbone_feature

        cls_out = fluid.layers.unsqueeze(backbone_feature, axes=[-1])
        reg_out = cls_out

        # classification branch
        for i in range(self.cfg.RPN.CLS_FC.__len__()):
            cls_out = conv_bn(cls_out, self.cfg.RPN.CLS_FC[i], bn=self.cfg.RPN.USE_BN, name='rpn_cls_{}'.format(i))
            # if i == 0 and self.cfg.RPN.DP_RATIO > 0:
            #     cls_out = fluid.layers.dropout(cls_out, self.cfg.RPN.DP_RATIO)
        # cls_out = conv_bn(cls_out, 1, bn=False, act=None, name='rpn_cls_out')
        cls_out = fluid.layers.conv2d(cls_out,
                                      num_filters=1,
				      filter_size=1,
				      stride=1,
				      padding=0,
				      dilation=1,
                                      param_attr=ParamAttr(
                                          initializer=Constant(1.376),
                                          name='rpn_cls_out_conv_weights'),
                                      bias_attr=ParamAttr(name='rpn_cls_out_conv_bias',
                                                          initializer=Constant(6.213),
                                                          # initializer=Constant(-np.log(99))
                                                          ))
        cls_out = fluid.layers.squeeze(cls_out, axes=[1, 3])
        self.outputs['rpn_cls'] = cls_out

        # regression branch
        per_loc_bin_num = int(self.cfg.RPN.LOC_SCOPE / self.cfg.RPN.LOC_BIN_SIZE) * 2
        if self.cfg.RPN.LOC_XZ_FINE:
            reg_channel = per_loc_bin_num * 4 + self.cfg.RPN.NUM_HEAD_BIN * 2 + 3
        else:
            reg_channel = per_loc_bin_num * 2 + self.cfg.RPN.NUM_HEAD_BIN * 2 + 3
        reg_channel += 1  # reg y

        for i in range(self.cfg.RPN.REG_FC.__len__()):
            reg_out = conv_bn(reg_out, self.cfg.RPN.REG_FC[i], bn=self.cfg.RPN.USE_BN, name='rpn_reg_{}'.format(i))
            # if i == 0 and self.cfg.RPN.DP_RATIO > 0:
            #     reg_out = fluid.layers.dropout(reg_out, self.cfg.RPN.DP_RATIO)
        # reg_out = conv_bn(reg_out, reg_channel, bn=False, act=None, name='rpn_reg_out')
        reg_out = fluid.layers.conv2d(reg_out,
                                      num_filters=reg_channel,
				      filter_size=1,
				      stride=1,
				      padding=0,
				      dilation=1,
                                      param_attr=ParamAttr(name='rpn_reg_out_conv_weights',
                                                           # initializer=Normal(0, 0.001),
                                                           initializer=Constant(7.392),
                                                           ),
                                      bias_attr=ParamAttr(
                                          initializer=Constant(3.213),
                                          name='rpn_reg_out_conv_bias'))
        reg_out = fluid.layers.squeeze(reg_out, axes=[3])
        reg_out = fluid.layers.transpose(reg_out, [0, 2, 1])
        self.outputs['rpn_reg'] = reg_out

        if self.mode != 'TRAIN' or self.cfg.RCNN.ENABLED:
            rpn_scores_row = cls_out
            rpn_scores_norm = fluid.layers.sigmoid(rpn_scores_row)
            seg_mask = fluid.layers.cast(rpn_scores_norm > self.cfg.RPN.SCORE_THRESH, dtype='float32')
            seg_mask = fluid.layers.cast(rpn_scores_norm > self.cfg.RPN.SCORE_THRESH, dtype='float32')
            pts_depth = fluid.layers.sqrt(fluid.layers.reduce_sum(backbone_xyz * backbone_xyz, dim=2))
            proposal_func = get_proposal_func(self.cfg, self.mode)
            proposal_input = fluid.layers.concat([fluid.layers.unsqueeze(rpn_scores_row, axes=[-1]),
                                                  backbone_xyz, reg_out], axis=-1)
            proposal = self.prog.current_block().create_var(name='proposal',
                                                            shape=[proposal_input.shape[1], 8],
                                                            dtype='float32')
            fluid.layers.py_func(proposal_func, proposal_input, proposal)
            rois, roi_scores_row = proposal[:, :, :7], proposal[:, :, -1]
            self.outputs['rois'] = rois
            self.outputs['roi_scores_row'] = roi_scores_row
            self.outputs['seg_mask'] = seg_mask
            self.outputs['pts_depth'] = pts_depth

    def get_outputs(self):
        return self.outputs

    def get_loss(self):
        assert self.inputs is not None, \
                "please call build() first"
        rpn_cls_label = self.inputs['rpn_cls_label']
        rpn_reg_label = self.inputs['rpn_reg_label']
        rpn_cls = self.outputs['rpn_cls']
        rpn_reg = self.outputs['rpn_reg']

        # RPN classification loss
        assert self.cfg.RPN.LOSS_CLS == "SigmoidFocalLoss", \
                "unsupported RPN cls loss type {}".format(self.cfg.RPN.LOSS_CLS)
        cls_flat = fluid.layers.reshape(rpn_cls, shape=[-1])
        cls_label_flat = fluid.layers.reshape(rpn_cls_label, shape=[-1])
        cls_label_pos = fluid.layers.cast(cls_label_flat > 0, dtype=cls_flat.dtype)
        pos_normalizer = fluid.layers.reduce_sum(cls_label_pos)
        cls_weights = fluid.layers.cast(cls_label_flat >= 0, dtype=cls_flat.dtype)
        cls_weights = cls_weights / fluid.layers.clip(pos_normalizer, min=1.0, max=1e10)
        cls_weights.stop_gradient = True
        cls_label_flat = fluid.layers.cast(cls_label_flat, dtype=cls_flat.dtype)
        cls_label_flat.stop_gradient = True
        rpn_loss_cls = sigmoid_focal_loss(cls_flat, cls_label_pos, cls_weights)
        rpn_loss_cls = fluid.layers.reduce_sum(rpn_loss_cls)

        # RPN regression loss
        rpn_reg = fluid.layers.reshape(rpn_reg, [-1, rpn_reg.shape[-1]])
        reg_label = fluid.layers.reshape(rpn_reg_label, [-1, rpn_reg_label.shape[-1]])
        fg_mask = fluid.layers.cast(cls_label_flat > 0, dtype=rpn_reg.dtype)
        fg_mask.stop_gradient = True
        loc_loss, angle_loss, size_loss, loss_dict = get_reg_loss(
                                        rpn_reg, reg_label, fg_mask,
                                        float(self.batch_size * self.cfg.RPN.NUM_POINTS),
                                        loc_scope=self.cfg.RPN.LOC_SCOPE,
                                        loc_bin_size=self.cfg.RPN.LOC_BIN_SIZE,
                                        num_head_bin=self.cfg.RPN.NUM_HEAD_BIN,
                                        anchor_size=self.cfg.CLS_MEAN_SIZE[0],
                                        get_xz_fine=self.cfg.RPN.LOC_XZ_FINE,
                                        get_y_by_bin=False,
                                        get_ry_fine=False)
        rpn_loss_reg = loc_loss + angle_loss + size_loss * 3

        self.rpn_loss = rpn_loss_cls * self.cfg.RPN.LOSS_WEIGHT[0] + rpn_loss_reg * self.cfg.RPN.LOSS_WEIGHT[1]
        return self.rpn_loss, rpn_loss_cls, rpn_loss_reg
        
if __name__ == "__main__":
    from utils.config import load_config, cfg
    batch_size = 4
    num_points = 16384
    load_config('./cfgs/default.yml')
    cfg.RPN.ENABLED = True
    # cfg.RCNN.ENABLED = False
    cfg.RCNN.ENABLED = True
    keys = ['sample_id', 'pts_input', 'rpn_cls_label', 'rpn_reg_label', 'pts_rect', 'pts_features', 'gt_boxes3d']
    # keys = ['pts_input', 'rpn_cls_label', 'rpn_reg_label']
    np_inputs = {}
    for key in keys:
        # np_inputs[key] = np.expand_dims(np.load('/paddle/rpn_data/{}.npy'.format(key)), axis=0)
        np_inputs[key] = np.load('/paddle/rpn_data/{}.npy'.format(key))[:batch_size]
        print(key, np_inputs[key].shape)

    sample_id = fluid.layers.data(name='sample_id', shape=[1], dtype='int32')
    pts_input = fluid.layers.data(name='pts_input', shape=[num_points, 3], dtype='float32', stop_gradient=False)
    pts_rect = fluid.layers.data(name='pts_rect', shape=[num_points, 3], dtype='float32')
    pts_features = fluid.layers.data(name='pts_features', shape=[num_points, 1], dtype='float32')
    rpn_cls_label = fluid.layers.data(name='rpn_cls_label', shape=[num_points], dtype='int32')
    rpn_reg_label = fluid.layers.data(name='rpn_reg_label', shape=[num_points, 7], dtype='float32')
    gt_boxes3d = fluid.layers.data(name='gt_boxes3d', shape=[100, 7], dtype='float32')
    inputs = {
            "pts_input": pts_input,
            "rpn_cls_label": rpn_cls_label,
            "rpn_reg_label": rpn_reg_label,
            }
    rpn = RPN(cfg, batch_size, num_points)
    rpn.build(inputs)
    rpn_loss, loss_cls, loss_reg = rpn.get_loss()
    optm = fluid.optimizer.Adam(learning_rate=0.01)
    optm.minimize(rpn_loss)

    place = fluid.CUDAPlace(0)
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())
    ret = exe.run(fetch_list=[
        rpn_loss.name,
        # loss_cls.name,
        # loss_reg.name,
        # rpn.backbone_xyz.name,
        # rpn.backbone_feature.name,
        rpn.outputs['rpn_cls'].name,
        rpn.outputs['rpn_reg'].name,
        'pts_input',
        rpn.outputs['backbone_feature'].name,
        rpn.outputs['rois'],
        rpn.outputs['roi_scores_row'],
        # 'sigmoid_cross_entropy_with_logits_0.tmp_0',
        # 'sigmoid_0.tmp_0',
        # 'cast_0.tmp_0',
        'sqrt_4.tmp_0',
        ], feed={
            'sample_id': np_inputs['sample_id'],
            'pts_input': np_inputs['pts_input'],
            'pts_rect': np_inputs['pts_rect'],
            'pts_features': np_inputs['pts_features'],
            'rpn_cls_label': np_inputs['rpn_cls_label'],
            'rpn_reg_label': np_inputs['rpn_reg_label'],
            'gt_boxes3d': np_inputs['gt_boxes3d'],
        })
    print(ret)
    np.save('rpn_cls.npy', ret[1])
    np.save('rpn_reg.npy', ret[2])
    np.save('pts_input.npy', ret[3])
    np.save('backbone_features.npy', ret[4])
    np.save('rois.npy', ret[5])
    np.save('roi_scores_row.npy', ret[6])
    np.save('pts_depth.npy', ret[7])
    # np.save('rpn_scores_norm.npy', ret[7])
    # np.save('seg_mask.npy', ret[8])
