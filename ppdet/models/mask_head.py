#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import paddle.fluid as fluid
import ppdet.models.backbones as backbones
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.initializer import Normal, Xavier, MSRA
from paddle.fluid.regularizer import L2Decay


class MaskHead(object):
    """
    MaskHead class
    """

    def __init__(
            self,
            cfg, ):
        """
        Args:
            cfg(Dict): All parameters in dictionary.
       """
        self.cfg = cfg
        self.roi_has_mask_int32 = None

    def get_target(self, rois, labels_int32, gt_label, is_crowd, gt_box,
                   im_info, gt_segms):
        """
        Get mask target for mask head loss.

        Args:
            rois(Variable), labels_int32(Variable): Output of 
                                generate_proposal_labels in bbox head.
            gt_label(Variable): Class label of groundtruth with shape [M, 1].
                                M is the number of groundtruth.
            is_crowd(Variable): A flag indicates whether a groundtruth is crowd
                                with shape [M, 1]. M is the number of 
                                groundtruth.
            gt_box(Variable): Bounding box in [xmin, ymin, xmax, ymax] format 
                              with shape [M, 4]. M is the number of groundtruth.
            im_info(Variable): A 2-D LoDTensor with shape [B, 3]. B is the 
                               number of input images, each element consists 
                               of im_height, im_width, im_scale.
            gt_segms(Variable): This input is a 2D LoDTensor with shape [S, 2],
                                it's LoD level is 3. The LoD[0] represents the 
                                gt objects number of each instance. LoD[1] 
                                represents the segmentation counts of each  
                                objects. LoD[2] represents the polygons number 
                                of each segmentation. S the total number of 
                                polygons coordinate points. Each element is 
                                (x, y) coordinate points."
 
        Returns:
            mask_rois(Variable): RoI with shape [P, 4], P is the number of mask.
                                 each element is a bounding box with 
                                 [xmin, ymin, xmax, ymax] format.
            roi_has_mask_int32(Variable): Output mask rois index with regard 
                                          to input rois with shape [P, 1], P is 
                                          the number of mask.
            mask_int32(Variable): a 4D LoDTensor with shape [P, Q], Q equal to 
                                  num_classes * resolution * resolution.
        """
        cfg = self.cfg
        mask_out = fluid.layers.generate_mask_labels(
            rois=rois,
            gt_classes=gt_label,
            is_crowd=is_crowd,
            gt_segms=gt_segms,
            im_info=im_info,
            labels_int32=labels_int32,
            num_classes=cfg.DATA.CLASS_NUM,
            resolution=cfg.MASK_HEAD.RESOLUTION)
        self.mask_rois = mask_out[0]
        self.roi_has_mask_int32 = mask_out[1]
        self.mask_int32 = mask_out[2]
        return self.mask_rois, self.roi_has_mask_int32, self.mask_int32

    def get_output(self, roi_feat, head_func, mode):
        """
        Get bbox head output.

        Args:
            roi_feat(Variable): RoI feature from RoIExtractor.
            head_func(Function): A function to extract mask_head feature.
            mode(String): Train or Test mode: 'train' or 'test'.

        Returns:
            mask_fcn_logits(Variable): Output of mask_head.
        """
        class_num = self.cfg.DATA.CLASS_NUM
        head_feat = head_func(self.cfg, roi_feat, self.roi_has_mask_int32)
        act_func = None
        if mode != 'train':
            act_func = 'sigmoid'
        mask_fcn_logits = fluid.layers.conv2d(
            input=head_feat,
            num_filters=class_num,
            filter_size=1,
            act=act_func,
            param_attr=ParamAttr(
                name='mask_fcn_logits_w', initializer=MSRA(uniform=False)),
            bias_attr=ParamAttr(
                name="mask_fcn_logits_b",
                learning_rate=2.,
                regularizer=L2Decay(0.)))
        return mask_fcn_logits

    def get_loss(self, mask_int32, mask_fcn_logits):
        """
        Get mask_head loss.

        Args:
            mask_int32(Variable): Output of get_target.
            mask_fcn_logits(Variable): Output of get_output.

        Return:
            loss_mask(Variable): mask_head loss.
        """
        class_num = self.cfg.DATA.CLASS_NUM
        resolution = self.cfg.MASK_HEAD.RESOLUTION
        mask_label = fluid.layers.cast(x=mask_int32, dtype='int64')
        mask_label.stop_gradient = True
        reshape_dim = class_num * resolution * resolution
        mask_fcn_logits_reshape = fluid.layers.reshape(mask_fcn_logits,
                                                       (-1, reshape_dim))
        loss_mask = fluid.layers.sigmoid_cross_entropy_with_logits(
            x=mask_fcn_logits_reshape,
            label=mask_label,
            ignore_index=-1,
            normalize=True)
        loss_mask = fluid.layers.reduce_sum(loss_mask, name='loss_mask')
        return loss_mask

    def get_prediction(self, bbox_det, roi_feat, head_func):
        """
        Get prediction mask in test stage.
        
        Args:
            bbox_det(Variable): Prediction bbox from bbox_head.get_prediction
                                with shape [N, 6]
            roi_feat(Variable): RoI feature from RoIExtractor.
            head_func(Function): A function to extract mask_head feature.
    
        Returns:
            mask_det(Variable): Prediction mask with shape 
                                [N, class_num, resolution, resolution]. 
        """
        mask_fcn_logits = self.get_output(roi_feat, head_func, 'test')
        mask_det = fluid.layers.lod_reset(mask_fcn_logits, bbox_det)
        return mask_det


def mask_v0upshare_train_head(cfg, roi_feat, index=None):
    return mask_v0upshare_head(cfg, roi_feat, 'train', index)


def mask_v0upshare_test_head(cfg, roi_feat, index=None):
    return mask_v0upshare_head(cfg, roi_feat, 'test')


def mask_v0upshare_head(cfg, roi_feat, mode, index=None):
    if mode == 'train':
        roi_feat = fluid.layers.gather(roi_feat, index)
    head_feat = fluid.layers.conv2d_transpose(
        input=roi_feat,
        num_filters=cfg.MASK_HEAD.DIM_REDUCED,
        filter_size=2,
        stride=2,
        act='relu',
        param_attr=ParamAttr(
            name='conv5_mask_w', initializer=MSRA(uniform=False)),
        bias_attr=ParamAttr(
            name='conv5_mask_b', learning_rate=2., regularizer=L2Decay(0.)))
    return head_feat


def mask_v1up4convs_head(cfg, roi_feat, index=None):
    return mask_v1upXconvs_head(cfg, roi_feat, 4)


def mask_v1upXconvs_head(cfg, roi_feat, num_convs):
    for i in range(num_convs):
        layer_name = "mask_inter_feat_" + str(i + 1)
        roi_feat = fluid.layers.conv2d(
            input=roi_feat,
            num_filters=cfg.MASK_HEAD.DIM_REDUCED,
            filter_size=3,
            padding=1 * cfg.MASK_HEAD.DILATION,
            act='relu',
            stride=1,
            dilation=cfg.MASK_HEAD.DILATION,
            name=layer_name,
            param_attr=ParamAttr(
                name=layer_name + '_w', initializer=MSRA(uniform=True)),
            bias_attr=ParamAttr(
                name=layer_name + '_b',
                learning_rate=2.,
                regularizer=L2Decay(0.)))
    head_feat = fluid.layers.conv2d_transpose(
        input=roi_feat,
        num_filters=cfg.dim_reduced,
        filter_size=2,
        stride=2,
        act='relu',
        param_attr=ParamAttr(
            name='conv5_mask_w', initializer=MSRA(uniform=False)),
        bias_attr=ParamAttr(
            name='conv5_mask_b', learning_rate=2., regularizer=L2Decay(0.)))
    return head_feat
