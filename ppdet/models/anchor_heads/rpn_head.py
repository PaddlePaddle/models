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

from ..registry import RPNHeads

__all__ = ['RPNHead']


@RPNHeads.register
class RPNHead(object):
    """
    RPNHead class
    
    Args:
        cfg(dict): All parameters in dictionary
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.is_train = cfg.IS_TRAIN
        # whether to use random to sample proposals.
        self.use_random = getattr(cfg.TRAIN, 'RANDOM', True)
        self.anchor = None
        self.anchor_var = None
        self.rpn_cls_score = None
        self.rpn_bbox_pred = None

    def _get_output(self, input):
        """
        Get anchor and RPN head output.

        Args:
            input(Variable): feature map from backbone with shape of [N, C, H, W]
        
        Returns:
            rpn_cls_score(Variable): Output of rpn head with shape of 
                [N, num_anchors, H, W].
            rpn_bbox_pred(Variable): Output of rpn head with shape of
                [N, num_anchors * 4, H, W].
        """
        dim_out = input.shape[1]
        rpn_conv = fluid.layers.conv2d(
            input=input,
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
        # Generate anchors
        self.anchor, self.anchor_var = fluid.layers.anchor_generator(
            input=rpn_conv,
            anchor_sizes=self.cfg.RPN_HEAD.ANCHOR.ANCHOR_SIZES,
            aspect_ratios=self.cfg.RPN_HEAD.ANCHOR.ASPECT_RATIOS,
            variance=self.cfg.RPN_HEAD.ANCHOR.VARIANCE,
            stride=self.cfg.RPN_HEAD.ANCHOR.RPN_STRIDE)
        num_anchor = self.anchor.shape[2]
        # Proposal classification scores
        self.rpn_cls_score = fluid.layers.conv2d(
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
        self.rpn_bbox_pred = fluid.layers.conv2d(
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
        return self.rpn_cls_score, self.rpn_bbox_pred

    def get_proposals(self, body_feats, im_info, body_feat_names):
        """
        Get proposals according to the output of backbone.

        Args:
            body_feats (Dict): The dictionary of feature maps from backbone.
            im_info(Variable): The information of image with shape [N, 3] with 
                shape (height, width, scale). 
            body_feat_names(List): A list of names of feature maps from 
                backbone.

        Returns:
            rpn_rois(Variable): Output proposals with shape of (rois_num, 4).
        """

        # In RPN Heads, only the last feature map of backbone is used.
        # And body_feat_names[-1] represents the last level name of backbone.
        body_feat = body_feats[body_feat_names[-1]]
        rpn_cls_score, rpn_bbox_pred = self._get_output(body_feat)

        rpn_cls_score_prob = fluid.layers.sigmoid(
            rpn_cls_score, name='rpn_cls_score_prob')

        rpn_cfg = self.cfg.RPN_HEAD.TRAIN if self.is_train else self.cfg.RPN_HEAD.TEST
        pre_nms_top_n = rpn_cfg.RPN_PRE_NMS_TOP_N
        post_nms_top_n = rpn_cfg.RPN_POST_NMS_TOP_N
        nms_thresh = rpn_cfg.RPN_NMS_THRESH
        min_size = rpn_cfg.RPN_MIN_SIZE
        eta = rpn_cfg.RPN_ETA
        rpn_rois, rpn_roi_probs = fluid.layers.generate_proposals(
            scores=rpn_cls_score_prob,
            bbox_deltas=rpn_bbox_pred,
            im_info=im_info,
            anchors=self.anchor,
            variances=self.anchor_var,
            pre_nms_top_n=pre_nms_top_n,
            post_nms_top_n=post_nms_top_n,
            nms_thresh=nms_thresh,
            min_size=min_size,
            eta=eta)
        return rpn_rois

    def _transform_input(self, rpn_cls_score, rpn_bbox_pred, anchor,
                         anchor_var):
        rpn_cls_score_reshape = fluid.layers.transpose(
            rpn_cls_score, perm=[0, 2, 3, 1])
        rpn_bbox_pred_reshape = fluid.layers.transpose(
            rpn_bbox_pred, perm=[0, 2, 3, 1])
        anchor_reshape = fluid.layers.reshape(anchor, shape=(-1, 4))
        var_reshape = fluid.layers.reshape(anchor_var, shape=(-1, 4))
        rpn_cls_score_reshape = fluid.layers.reshape(
            x=rpn_cls_score_reshape, shape=(0, -1, 1))
        rpn_bbox_pred_reshape = fluid.layers.reshape(
            x=rpn_bbox_pred_reshape, shape=(0, -1, 4))
        return rpn_cls_score_reshape, rpn_bbox_pred_reshape, anchor_reshape, var_reshape

    def _get_loss_input(self):
        if self.rpn_cls_score is None:
            raise ValueError("self.rpn_cls_score should be not None, "
                             "should call RPNHead.get_proposals at first")
        if self.rpn_bbox_pred is None:
            raise ValueError("self.rpn_bbox_pred should be not None, "
                             "should call RPNHead.get_proposals at first")
        if self.anchor is None:
            raise ValueError("self.anchor should be not None, "
                             "should call RPNHead.get_proposals at first")
        if self.anchor_var is None:
            raise ValueError("self.anchor_var should be not None, "
                             "should call RPNHead.get_proposals at first")

        loss_input = self._transform_input(self.rpn_cls_score,
                                           self.rpn_bbox_pred, self.anchor,
                                           self.anchor_var)
        return loss_input

    def get_loss(self, im_info, gt_box, is_crowd):
        """
        Sample proposals and Calculate rpn loss.

        Args:
            im_info(Variable): The information of image with shape [N, 3] with
                shape (height, width, scale). 
            gt_box(Variable): The ground-truth bounding boxes with shape [M, 4].
                M is the number of groundtruth.
            is_crowd(Variable): Indicates groud-truth is crowd or not with
                shape [M, 1]. M is the number of groundtruth.

        Returns:
            Type: Dict 
                rpn_cls_loss(Variable): RPN classification loss.
                rpn_bbox_loss(Variable): RPN bounding box regression loss. 
            
        """
        rpn_cls, rpn_bbox, anchor, anchor_var = self._get_loss_input()
        score_pred, loc_pred, score_tgt, loc_tgt, bbox_weight = \
            fluid.layers.rpn_target_assign(
                bbox_pred=rpn_bbox,
                cls_logits=rpn_cls,
                anchor_box=anchor,
                anchor_var=anchor_var,
                gt_boxes=gt_box,
                is_crowd=is_crowd,
                im_info=im_info,
                rpn_batch_size_per_im=self.cfg.RPN_HEAD.RPN_BATCH_SIZE_PER_IM,
                rpn_straddle_thresh=self.cfg.RPN_HEAD.RPN_STRADDLE_THRESH,
                rpn_fg_fraction=self.cfg.RPN_HEAD.RPN_FG_FRACTION,
                rpn_positive_overlap=self.cfg.RPN_HEAD.RPN_POSITIVE_OVERLAP,
                rpn_negative_overlap=self.cfg.RPN_HEAD.RPN_NEGATIVE_OVERLAP,
                use_random=self.use_random)

        score_tgt = fluid.layers.cast(x=score_tgt, dtype='float32')
        score_tgt.stop_gradient = True
        rpn_cls_loss = fluid.layers.sigmoid_cross_entropy_with_logits(
            x=score_pred, label=score_tgt)
        rpn_cls_loss = fluid.layers.reduce_mean(
            rpn_cls_loss, name='loss_rpn_cls')

        loc_tgt = fluid.layers.cast(x=loc_tgt, dtype='float32')
        loc_tgt.stop_gradient = True
        rpn_reg_loss = fluid.layers.smooth_l1(
            x=loc_pred,
            y=loc_tgt,
            sigma=3.0,
            inside_weight=bbox_weight,
            outside_weight=bbox_weight)
        rpn_reg_loss = fluid.layers.reduce_sum(
            rpn_reg_loss, name='loss_rpn_bbox')
        score_shape = fluid.layers.shape(score_tgt)
        score_shape = fluid.layers.cast(x=score_shape, dtype='float32')
        norm = fluid.layers.reduce_prod(score_shape)
        norm.stop_gradient = True
        rpn_reg_loss = rpn_reg_loss / norm

        return {'loss_rpn_cls': rpn_cls_loss, 'loss_rpn_bbox': rpn_reg_loss}
