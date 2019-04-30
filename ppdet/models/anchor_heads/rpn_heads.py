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
from __future__ import unicode_literals

import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.initializer import Normal
from paddle.fluid.regularizer import L2Decay

__all__ = ['RPNHead']


class RPNHead(object):
    def __init__(self,
                 cfg,
                 gt_box,
                 gt_label,
                 is_crowd,
                 im_info,
                 gt_masks=None,
                 mode='train',
                 use_random=True):
        self.cfg = cfg
        self.gt_label = gt_label
        self.gt_box = gt_box
        self.is_crowd = is_crowd
        self.im_info = im_info
        self.gt_masks = gt_masks
        self.use_random = use_random
        self.mode = mode

    def get_output(self, rpn_input):
        dim_out = rpn_input.shape[1]
        rpn_conv = fluid.layers.conv2d(
            input=rpn_input,
            num_filters=dim_out,
            filter_size=3,
            stride=1,
            padding=1,
            act='relu',
            name='conv_rpn',
            param_attr=ParamAttr(
                name="conv_rpn_w", initializer=Normal(
                    loc=0., scale=0.01)),
            bias_attr=ParamAttr(
                name="conv_rpn_b", learning_rate=2., regularizer=L2Decay(0.)))
        anchor, var = fluid.layers.anchor_generator(
            input=rpn_conv,
            anchor_sizes=self.cfg.RPN_HEAD.ANCHOR.ANCHOR_SIZES,
            aspect_ratios=self.cfg.RPN_HEAD.ANCHOR.ASPECT_RATIOS,
            variance=self.cfg.RPN_HEAD.ANCHOR.VARIANCE,
            stride=self.cfg.RPN_HEAD.ANCHOR.RPN_STRIDE)
        num_anchor = anchor.shape[2]
        # Proposal classification scores
        rpn_cls_score = fluid.layers.conv2d(
            rpn_conv,
            num_filters=num_anchor,
            filter_size=1,
            stride=1,
            padding=0,
            act=None,
            name='rpn_cls_score',
            param_attr=ParamAttr(
                name="rpn_cls_logits_w", initializer=Normal(
                    loc=0., scale=0.01)),
            bias_attr=ParamAttr(
                name="rpn_cls_logits_b",
                learning_rate=2.,
                regularizer=L2Decay(0.)))
        # Proposal bbox regression deltas
        rpn_bbox_pred = fluid.layers.conv2d(
            rpn_conv,
            num_filters=4 * num_anchor,
            filter_size=1,
            stride=1,
            padding=0,
            act=None,
            name='rpn_bbox_pred',
            param_attr=ParamAttr(
                name="rpn_bbox_pred_w", initializer=Normal(
                    loc=0., scale=0.01)),
            bias_attr=ParamAttr(
                name="rpn_bbox_pred_b",
                learning_rate=2.,
                regularizer=L2Decay(0.)))
        return anchor, var, rpn_cls_score, rpn_bbox_pred

    def get_poposals(self, rpn_cls_score, rpn_bbox_pred, anchor, var):
        rpn_cls_score_prob = fluid.layers.sigmoid(
            rpn_cls_score, name='rpn_cls_score_prob')

        param_obj = self.cfg.TRAIN if self.mode == 'train' else self.cfg.TEST
        pre_nms_top_n = param_obj.RPN_PRE_NMS_TOP_N
        post_nms_top_n = param_obj.RPN_POST_NMS_TOP_N
        nms_thresh = param_obj.RPN_NMS_THRESH
        min_size = param_obj.RPN_MIN_SIZE
        eta = param_obj.RPN_ETA
        rpn_rois, rpn_roi_probs = fluid.layers.generate_proposals(
            scores=rpn_cls_score_prob,
            bbox_deltas=rpn_bbox_pred,
            im_info=self.im_info,
            anchors=anchor,
            variances=var,
            pre_nms_top_n=pre_nms_top_n,
            post_nms_top_n=post_nms_top_n,
            nms_thresh=nms_thresh,
            min_size=min_size,
            eta=eta)
        return rpn_rois, rpn_roi_probs

    def get_proposal_targets(self, rpn_rois):
        # rois, labels_int32, bbox_targets, bbox_inside_weights, bbox_outside_weights
        target_outs = fluid.layers.generate_proposal_labels(
            rpn_rois=rpn_rois,
            gt_classes=self.gt_label,
            is_crowd=self.is_crowd,
            gt_boxes=self.gt_box,
            im_info=self.im_info,
            batch_size_per_im=self.cfg.RPN_HEAD.PROPOSAL.BATCH_SIZE_PER_IM,
            fg_fraction=self.cfg.RPN_HEAD.PROPOSAL.FG_FRACTION,
            fg_thresh=self.cfg.RPN_HEAD.PROPOSAL.FG_THRESH,
            bg_thresh_hi=self.cfg.RPN_HEAD.PROPOSAL.BG_THRESH_HI,
            bg_thresh_lo=self.cfg.RPN_HEAD.PROPOSAL.BG_THRESH_LO,
            bbox_reg_weights=self.cfg.RPN_HEAD.PROPOSAL.BBOX_REG_WEIGHTS,
            class_nums=self.cfg.DATA.CLASS_NUM,
            use_random=self.use_random)

        rois = target_outs[0]
        labels_int32 = target_outs[1]
        bbox_targets = target_outs[2]
        bbox_inside_weights = target_outs[3]
        bbox_outside_weights = target_outs[4]
        return rois, labels_int32, bbox_targets, bbox_inside_weights, bbox_outside_weights

    def get_mask_targets(self, rois, labels_int32):
        mask_target_outs = fluid.layers.generate_mask_labels(
            im_info=self.im_info,
            gt_classes=self.gt_label,
            is_crowd=self.is_crowd,
            gt_segms=self.gt_masks,
            rois=rois,
            labels_int32=labels_int32,
            num_classes=self.cfg.DATA.CLASS_NUM,
            resolution=self.cfg.RPN_HEAD.RESOLUTION)
        mask_rois = mask_target_outs[0]
        roi_has_mask_int32 = mask_target_outs[1]
        mask_int32 = mask_target_outs[2]
        return mask_rois, roi_has_mask_int32, mask_int32

    def _get_rpn_loss_input(self, rpn_cls_score, rpn_bbox_pred, anchor, var):
        rpn_cls_score_reshape = fluid.layers.transpose(
            rpn_cls_score, perm=[0, 2, 3, 1])
        rpn_bbox_pred_reshape = fluid.layers.transpose(
            rpn_bbox_pred, perm=[0, 2, 3, 1])

        anchor_reshape = fluid.layers.reshape(anchor, shape=(-1, 4))
        var_reshape = fluid.layers.reshape(var, shape=(-1, 4))

        rpn_cls_score_reshape = fluid.layers.reshape(
            x=rpn_cls_score_reshape, shape=(0, -1, 1))
        rpn_bbox_pred_reshape = fluid.layers.reshape(
            x=rpn_bbox_pred_reshape, shape=(0, -1, 4))
        score_pred, loc_pred, score_tgt, loc_tgt, bbox_weight = \
            fluid.layers.rpn_target_assign(
                bbox_pred=rpn_bbox_pred_reshape,
                cls_logits=rpn_cls_score_reshape,
                anchor_box=anchor_reshape,
                anchor_var=var_reshape,
                gt_boxes=self.gt_box,
                is_crowd=self.is_crowd,
                im_info=self.im_info,
                rpn_batch_size_per_im=self.cfg.RPN_HEAD.RPN_BATCH_SIZE_PER_IM,
                rpn_straddle_thresh=self.cfg.RPN_HEAD.RPN_STRADDLE_THRESH,
                rpn_fg_fraction=self.cfg.RPN_HEAD.RPN_FG_FRACTION,
                rpn_positive_overlap=self.cfg.RPN_HEAD.RPN_POSITIVE_OVERLAP,
                rpn_negative_overlap=self.cfg.RPN_HEAD.RPN_NEGATIVE_OVERLAP,
                use_random=self.use_random)
        return score_pred, loc_pred, score_tgt, loc_tgt, bbox_weight

    def get_loss(self, rpn_cls_score, rpn_bbox_pred, anchor, var):
        score_pred, loc_pred, score_tgt, loc_tgt, bbox_weight = self._get_rpn_loss_input(
            rpn_cls_score, rpn_bbox_pred, anchor, var)
        score_tgt = fluid.layers.cast(x=score_tgt, dtype='float32')
        score_tgt.stop_gradient = True
        rpn_cls_loss = fluid.layers.sigmoid_cross_entropy_with_logits(
            x=score_pred, label=score_tgt)

        rpn_cls_loss = fluid.layers.reduce_sum(rpn_cls_loss)
        rpn_cls_loss = rpn_cls_loss / (self.cfg.TRAIN.IM_PER_BATCH *
                                       self.cfg.RPN_HEAD.RPN_BATCH_SIZE_PER_IM)

        loc_tgt = fluid.layers.cast(x=loc_tgt, dtype='float32')
        loc_tgt.stop_gradient = True
        rpn_bbox_loss = fluid.layers.smooth_l1(
            x=loc_pred,
            y=loc_tgt,
            sigma=3.0,
            inside_weight=bbox_weight,
            outside_weight=bbox_weight)

        rpn_bbox_loss = fluid.layers.reduce_sum(rpn_bbox_loss)
        rpn_bbox_loss = rpn_bbox_loss / (
            self.cfg.TRAIN.IM_PER_BATCH *
            self.cfg.RPN_HEAD.RPN_BATCH_SIZE_PER_IM)

        return rpn_cls_loss, rpn_bbox_loss
