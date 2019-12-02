# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

from paddle import fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.initializer import Normal
from paddle.fluid.regularizer import L2Decay

from ppdet.core.workspace import register
from ppdet.modeling.ops import (AnchorGenerator, RPNTargetAssign,
                                GenerateProposals, RotatedAnchorGenerator,
                                RRPNTargetAssign, RotatedGenerateProposals)

__all__ = ['RRPNTargetAssign', 'RotatedGenerateProposals', 'RRPNHead']


@register
class RRPNHead(object):
    """
    RPN Head

    Args:
        anchor_generator (object): `AnchorGenerator` instance
        rpn_target_assign (object): `RPNTargetAssign` instance
        train_proposal (object): `GenerateProposals` instance for training
        test_proposal (object): `GenerateProposals` instance for testing
        num_classes (int): number of classes in rpn output
    """
    __inject__ = [
        'rotated_anchor_generator', 'rrpn_target_assign', 'train_proposal',
        'test_proposal'
    ]

    def __init__(self,
                 rotated_anchor_generator=RotatedAnchorGenerator().__dict__,
                 rrpn_target_assign=RRPNTargetAssign().__dict__,
                 train_proposal=RotatedGenerateProposals(12000, 2000).__dict__,
                 test_proposal=RotatedGenerateProposals().__dict__,
                 num_classes=1):
        super(RRPNHead, self).__init__()
        self.anchor_generator = rotated_anchor_generator
        self.rpn_target_assign = rrpn_target_assign
        self.train_proposal = train_proposal
        self.test_proposal = test_proposal
        self.num_classes = num_classes
        if isinstance(rotated_anchor_generator, dict):
            self.anchor_generator = RotatedAnchorGenerator(
                **rotated_anchor_generator)
        if isinstance(rrpn_target_assign, dict):
            self.rpn_target_assign = RRPNTargetAssign(**rrpn_target_assign)
        if isinstance(train_proposal, dict):
            self.train_proposal = RotatedGenerateProposals(**train_proposal)
        if isinstance(test_proposal, dict):
            self.test_proposal = RotatedGenerateProposals(**test_proposal)

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
        self.anchor, self.anchor_var = self.anchor_generator(input=rpn_conv)
        num_anchor = self.anchor.shape[2]
        # Proposal classification scores
        self.rpn_cls_score = fluid.layers.conv2d(
            rpn_conv,
            num_filters=num_anchor * self.num_classes,
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
            num_filters=5 * num_anchor,
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

    def get_proposals(self, body_feats, im_info, mode='train'):
        """
        Get proposals according to the output of backbone.

        Args:
            body_feats (dict): The dictionary of feature maps from backbone.
            im_info(Variable): The information of image with shape [N, 3] with
                shape (height, width, scale).
            body_feat_names(list): A list of names of feature maps from
                backbone.

        Returns:
            rpn_rois(Variable): Output proposals with shape of (rois_num, 4).
        """

        # In RPN Heads, only the last feature map of backbone is used.
        # And body_feat_names[-1] represents the last level name of backbone.
        body_feat = list(body_feats.values())[-1]
        rpn_cls_score, rpn_bbox_pred = self._get_output(body_feat)

        if self.num_classes == 1:
            rpn_cls_prob = fluid.layers.sigmoid(
                rpn_cls_score, name='rpn_cls_prob')
        else:
            rpn_cls_score = fluid.layers.transpose(
                rpn_cls_score, perm=[0, 2, 3, 1])
            rpn_cls_score = fluid.layers.reshape(
                rpn_cls_score, shape=(0, 0, 0, -1, self.num_classes))
            rpn_cls_prob_tmp = fluid.layers.softmax(
                rpn_cls_score, use_cudnn=False, name='rpn_cls_prob')
            rpn_cls_prob_slice = fluid.layers.slice(
                rpn_cls_prob_tmp, axes=[4], starts=[1],
                ends=[self.num_classes])
            rpn_cls_prob, _ = fluid.layers.topk(rpn_cls_prob_slice, 1)
            rpn_cls_prob = fluid.layers.reshape(
                rpn_cls_prob, shape=(0, 0, 0, -1))
            rpn_cls_prob = fluid.layers.transpose(
                rpn_cls_prob, perm=[0, 3, 1, 2])
        prop_op = self.train_proposal if mode == 'train' else self.test_proposal
        rpn_rois, rpn_roi_probs = prop_op(
            scores=rpn_cls_prob,
            bbox_deltas=rpn_bbox_pred,
            im_info=im_info,
            anchors=self.anchor,
            variances=self.anchor_var)
        return rpn_rois

    def _transform_input(self, rpn_cls_score, rpn_bbox_pred, anchor,
                         anchor_var):
        rpn_cls_score = fluid.layers.transpose(rpn_cls_score, perm=[0, 2, 3, 1])
        rpn_bbox_pred = fluid.layers.transpose(rpn_bbox_pred, perm=[0, 2, 3, 1])
        anchor = fluid.layers.reshape(anchor, shape=(-1, 5))
        anchor_var = fluid.layers.reshape(anchor_var, shape=(-1, 5))
        rpn_cls_score = fluid.layers.reshape(
            x=rpn_cls_score, shape=(0, -1, self.num_classes))
        rpn_bbox_pred = fluid.layers.reshape(x=rpn_bbox_pred, shape=(0, -1, 5))
        return rpn_cls_score, rpn_bbox_pred, anchor, anchor_var

    def _get_loss_input(self):
        for attr in ['rpn_cls_score', 'rpn_bbox_pred', 'anchor', 'anchor_var']:
            if not getattr(self, attr, None):
                raise ValueError("self.{} should not be None,".format(attr),
                                 "call RPNHead.get_proposals first")
        return self._transform_input(self.rpn_cls_score, self.rpn_bbox_pred,
                                     self.anchor, self.anchor_var)

    def get_loss(self, im_info, gt_box, is_crowd, gt_label=None):
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
            Type: dict
                rpn_cls_loss(Variable): RPN classification loss.
                rpn_bbox_loss(Variable): RPN bounding box regression loss.

        """
        rpn_cls, rpn_bbox, anchor, anchor_var = self._get_loss_input()
        if self.num_classes == 1:
            score_pred, loc_pred, score_tgt, loc_tgt = \
                self.rpn_target_assign(
                    bbox_pred=rpn_bbox,
                    cls_logits=rpn_cls,
                    anchor_box=anchor,
                    #anchor_var=anchor_var,
                    gt_boxes=gt_box,
                    #is_crowd=is_crowd,
                    im_info=im_info)
            score_tgt = fluid.layers.cast(x=score_tgt, dtype='float32')
            score_tgt.stop_gradient = True
            rpn_cls_loss = fluid.layers.sigmoid_cross_entropy_with_logits(
                x=score_pred, label=score_tgt)
        else:
            score_pred, loc_pred, score_tgt, loc_tgt = \
                self.rpn_target_assign(
                    bbox_pred=rpn_bbox,
                    cls_logits=rpn_cls,
                    anchor_box=anchor,
                    anchor_var=anchor_var,
                    gt_boxes=gt_box,
                    gt_labels=gt_label,
                    is_crowd=is_crowd,
                    num_classes=self.num_classes,
                    im_info=im_info)
            labels_int64 = fluid.layers.cast(x=score_tgt, dtype='int64')
            labels_int64.stop_gradient = True
            rpn_cls_loss = fluid.layers.softmax_with_cross_entropy(
                logits=score_pred, label=labels_int64, numeric_stable_mode=True)

        rpn_cls_loss = fluid.layers.reduce_mean(
            rpn_cls_loss, name='loss_rpn_cls')

        loc_tgt = fluid.layers.cast(x=loc_tgt, dtype='float32')
        loc_tgt.stop_gradient = True
        rpn_reg_loss = fluid.layers.smooth_l1(
            x=loc_pred, y=loc_tgt, sigma=3.0)  #,
        rpn_reg_loss = fluid.layers.reduce_sum(
            rpn_reg_loss, name='loss_rpn_bbox')
        score_shape = fluid.layers.shape(score_tgt)
        score_shape = fluid.layers.cast(x=score_shape, dtype='float32')
        norm = fluid.layers.reduce_prod(score_shape)
        norm.stop_gradient = True
        rpn_reg_loss = rpn_reg_loss / norm

        return {'loss_rpn_cls': rpn_cls_loss, 'loss_rpn_bbox': rpn_reg_loss}
