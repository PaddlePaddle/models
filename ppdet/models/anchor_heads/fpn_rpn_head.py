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
from ppdet.models.anchor_heads.rpn_head import RPNHead

from ..registry import RPNHeads

__all__ = ['FPNRPNHead']


@RPNHeads.register
class FPNRPNHead(RPNHead):
    """
    FPNRPNHead class
    
    Args:
        cfg(dict): All parameters in dictionary

    TODO(guanzhong): add more detailed comments.
    """

    def __init__(self, cfg):
        super(FPNRPNHead, self).__init__(cfg)
        self.fpn_dim = cfg.FPN.DIM
        self.k_max = cfg.FPN.RPN_MAX_LEVEL
        self.k_min = cfg.FPN.RPN_MIN_LEVEL
        self.fpn_rpn_list = []
        self.anchors_list = []
        self.anchor_var_list = []

    def _get_output(self, input, feat_lvl):
        """
        Get anchor and FPN RPN head output at one level.
        
        Args:
            input(Variable): Body feature from backbone.
            feat_lvl(Integer): Indicate the level of rpn output corresponding
                to the level of feature map.

        Return:
            rpn_cls_score(Variable): Output of one level of fpn rpn head with 
                shape of [N, num_anchors, H, W].
            rpn_bbox_pred(Variable): Output of one level of fpn rpn head with 
                shape of [N, num_anchors * 4, H, W].
        """
        fpn_cfg = self.cfg.FPN
        slvl = str(feat_lvl)
        conv_name = 'conv_rpn_fpn' + slvl
        cls_name = 'rpn_cls_logits_fpn' + slvl
        bbox_name = 'rpn_bbox_pred_fpn' + slvl
        conv_share_name = 'conv_rpn_fpn' + str(self.k_min)
        cls_share_name = 'rpn_cls_logits_fpn' + str(self.k_min)
        bbox_share_name = 'rpn_bbox_pred_fpn' + str(self.k_min)

        num_anchors = len(fpn_cfg.RPN_ASPECT_RATIOS)
        conv_rpn_fpn = fluid.layers.conv2d(
            input=input,
            num_filters=self.fpn_dim,
            filter_size=3,
            padding=1,
            act='relu',
            name=conv_name,
            param_attr=ParamAttr(
                name=conv_share_name + '_w',
                initializer=Normal(
                    loc=0., scale=0.01)),
            bias_attr=ParamAttr(
                name=conv_share_name + '_b',
                learning_rate=2.,
                regularizer=L2Decay(0.)))

        self.anchors, self.anchor_var = fluid.layers.anchor_generator(
            input=conv_rpn_fpn,
            anchor_sizes=(fpn_cfg.RPN_ANCHOR_START_SIZE * 2.
                          **(feat_lvl - self.k_min), ),
            aspect_ratios=fpn_cfg.RPN_ASPECT_RATIOS,
            variance=self.cfg.RPN_HEAD.ANCHOR.VARIANCE,
            stride=(2.**feat_lvl, 2.**feat_lvl))

        self.rpn_cls_score = fluid.layers.conv2d(
            input=conv_rpn_fpn,
            num_filters=num_anchors,
            filter_size=1,
            act=None,
            name=cls_name,
            param_attr=ParamAttr(
                name=cls_share_name + '_w',
                initializer=Normal(
                    loc=0., scale=0.01)),
            bias_attr=ParamAttr(
                name=cls_share_name + '_b',
                learning_rate=2.,
                regularizer=L2Decay(0.)))
        self.rpn_bbox_pred = fluid.layers.conv2d(
            input=conv_rpn_fpn,
            num_filters=num_anchors * 4,
            filter_size=1,
            act=None,
            name=bbox_name,
            param_attr=ParamAttr(
                name=bbox_share_name + '_w',
                initializer=Normal(
                    loc=0., scale=0.01)),
            bias_attr=ParamAttr(
                name=bbox_share_name + '_b',
                learning_rate=2.,
                regularizer=L2Decay(0.)))
        return self.rpn_cls_score, self.rpn_bbox_pred

    def _get_single_proposals(self, body_feat, im_info, feat_lvl):
        """
        Get proposals in one level according to the output of fpn rpn head

        Args:
            body_feat(Variable): the feature map from backone.
            im_info(Variable): The information of image with shape [N, 3] with 
                format (height, width, scale).  
            feat_lvl(Integer): Indicate the level of proposals corresponding to
                the feature maps.

        Returns:
            rpn_rois_fpn(Variable): Output proposals with shape of (rois_num, 4).
            rpn_roi_probs_fpn(Variable): Scores of proposals with 
                shape of (rois_num, 1).
        """

        rpn_cls_logits_fpn, rpn_bbox_pred_fpn = self._get_output(body_feat,
                                                                 feat_lvl)

        rpn_cls_prob_fpn = fluid.layers.sigmoid(
            rpn_cls_logits_fpn, name='rpn_cls_probs_fpn' + str(feat_lvl))

        rpn_cfg = self.cfg.RPN_HEAD.TRAIN if self.is_train else self.cfg.RPN_HEAD.TEST
        pre_nms_top_n = rpn_cfg.RPN_PRE_NMS_TOP_N
        post_nms_top_n = rpn_cfg.RPN_POST_NMS_TOP_N
        nms_thresh = rpn_cfg.RPN_NMS_THRESH
        min_size = rpn_cfg.RPN_MIN_SIZE
        eta = rpn_cfg.RPN_ETA
        rpn_rois_fpn, rpn_roi_probs_fpn = fluid.layers.generate_proposals(
            scores=rpn_cls_prob_fpn,
            bbox_deltas=rpn_bbox_pred_fpn,
            im_info=im_info,
            anchors=self.anchors,
            variances=self.anchor_var,
            pre_nms_top_n=pre_nms_top_n,
            post_nms_top_n=post_nms_top_n,
            nms_thresh=nms_thresh,
            min_size=min_size,
            eta=eta)
        return rpn_rois_fpn, rpn_roi_probs_fpn

    def get_proposals(self, fpn_feats, im_info, fpn_feat_names):
        """
        Get proposals in multiple levels according to the output of fpn 
        rpn head

        Args:
            fpn_feats(Dict): A dictionary represents the output feature map
                of FPN neck with their name.
            im_info(Variable): The information of image with shape [N, 3] with
                format (height, width, scale).
            fpn_feat_names(List): A list of names regarding to output of FPN neck.

        Return:
            rois_list(Variable): Output proposals in shape of [rois_num, 4]
        """
        rois_list = []
        roi_probs_list = []
        num_anchors = len(self.cfg.FPN.RPN_ASPECT_RATIOS)
        for lvl in range(self.k_min, self.k_max + 1):
            fpn_feat_name = fpn_feat_names[self.k_max - lvl]
            fpn_feat = fpn_feats[fpn_feat_name]
            rois_fpn, roi_probs_fpn = self._get_single_proposals(fpn_feat,
                                                                 im_info, lvl)
            self.fpn_rpn_list.append((self.rpn_cls_score, self.rpn_bbox_pred))
            rois_list.append(rois_fpn)
            roi_probs_list.append(roi_probs_fpn)
            self.anchors_list.append(self.anchors)
            self.anchor_var_list.append(self.anchor_var)
        collect_cfg = self.cfg.RPN_HEAD.TRAIN if self.is_train else self.cfg.RPN_HEAD.TEST
        post_nms_top_n = collect_cfg.RPN_POST_NMS_TOP_N
        rois_collect = fluid.layers.collect_fpn_proposals(
            rois_list,
            roi_probs_list,
            self.k_min,
            self.k_max,
            post_nms_top_n,
            name='collect')
        return rois_collect

    def _get_loss_input(self):
        rpn_cls_reshape_list = []
        rpn_bbox_reshape_list = []
        anchors_reshape_list = []
        anchor_var_reshape_list = []
        for i in range(len(self.fpn_rpn_list)):
            single_input = self._transform_input(
                self.fpn_rpn_list[i][0], self.fpn_rpn_list[i][1],
                self.anchors_list[i], self.anchor_var_list[i])
            rpn_cls_reshape_list.append(single_input[0])
            rpn_bbox_reshape_list.append(single_input[1])
            anchors_reshape_list.append(single_input[2])
            anchor_var_reshape_list.append(single_input[3])

        rpn_cls_input = fluid.layers.concat(rpn_cls_reshape_list, axis=1)
        rpn_bbox_input = fluid.layers.concat(rpn_bbox_reshape_list, axis=1)
        anchors_input = fluid.layers.concat(anchors_reshape_list)
        anchor_var_input = fluid.layers.concat(anchor_var_reshape_list)
        return rpn_cls_input, rpn_bbox_input, anchors_input, anchor_var_input
