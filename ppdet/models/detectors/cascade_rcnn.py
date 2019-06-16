"""
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
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import six

import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.framework import Variable

from ..registry import Detectors
from ..registry import Backbones
from ..registry import RPNHeads
from ..registry import RoIExtractors
from ..registry import Necks
from ..target_assigners.bbox_assigner import CascadeBBoxAssigner

from .faster_rcnn import FasterRCNN

__all__ = ['CascadeRCNN']


@Detectors.register
class CascadeRCNN(FasterRCNN):
    """
    CascadeRCNN class
    Args:
        cfg (Dict): All parameters in dictionary.
    """

    def __init__(self, cfg):
        super(CascadeRCNN, self).__init__(cfg)
        self.bbox_assigner = CascadeBBoxAssigner(cfg)
        brw = self.cfg.RPN_HEAD.PROPOSAL.BBOX_REG_WEIGHTS
        brw0, brw1, brw2 = brw[0], brw[1], brw[2]
        self.cascade_bbox_reg_weights = [
            [1., 1., 1., 1.], [1. / brw0, 1. / brw0, 2. / brw0, 2. / brw0],
            [1. / brw1, 1. / brw1, 2. / brw1, 2. / brw1],
            [1. / brw2, 1. / brw2, 2. / brw2, 2. / brw2]
        ]
        self.is_cls_agnostic = self.cfg.RPN_HEAD.PROPOSAL.CLS_AGNOSTIC_BBOX_REG
        self.cls_agnostic_bbox_reg = 2 if self.is_cls_agnostic \
     else self.cfg.DATA.CLASS_NUM
        self.cascade_rcnn_loss_weight = self.cfg.TRAIN.CASCADE_LOSS_WEIGHT

    def _forward(self):
        # inputs
        feed_vars = self.build_feeds(self.feed_info(), self.use_pyreader)
        im = feed_vars['image']
        im_info = feed_vars['im_info']
        if self.is_train:
            gt_box = feed_vars['gt_box']
            is_crowd = feed_vars['is_crowd']

        # backbone
        body_dict = self.backbone(im)
        body_name_list = self.backbone.get_body_feat_names()

        # fpn
        fpn_dict, spatial_scale, fpn_name_list = self.neck.get_output(
            body_dict, body_name_list)

        # rpn proposals
        rpn_rois = self.rpn_head.get_proposals(fpn_dict, im_info, fpn_name_list)

        if self.is_train:
            rpn_loss = self.rpn_head.get_loss(im_info, gt_box, is_crowd)

        proposal_list = []
        roi_feat_list = []
        rcnn_pred_list = []
        rcnn_target_list = []
        for i in range(3):
            if i > 0:
                refined_bbox = self._decode_box(
                    proposals,
                    bbox_pred,
                    self.cascade_bbox_reg_weights[i], )
            else:
                refined_bbox = rpn_rois

            if self.is_train:
                outs = self.bbox_assigner.get_sampled_rois_and_targets(
                    refined_bbox,
                    feed_vars,
                    is_cls_agnostic=self.is_cls_agnostic,
                    is_cascade_rcnn=True if i > 0 else False,
                    cascade_curr_stage=i, )
                proposals = outs[0]
                rcnn_target_list.append(outs)
            else:
                proposals = refined_bbox
            proposal_list.append(proposals)

            # RoI Extractor
            roi_feat = self.roi_extractor.get_roi_feat(
                fpn_dict, proposals, fpn_name_list, spatial_scale)
            roi_feat_list.append(roi_feat)
            # fast-rcnn head
            cls_score, bbox_pred = self.bbox_head.get_output(
                roi_feat,
                wb_scalar=1.0 / self.cascade_rcnn_loss_weight[i],
                name='_' + str(i + 1) if i > 0 else '')
            rcnn_pred_list.append((cls_score, bbox_pred))

        if self.is_train:
            loss = self.bbox_head.get_loss(rcnn_pred_list, rcnn_target_list,
                                           self.cascade_rcnn_loss_weight)
            loss.update(rpn_loss)
            total_loss = fluid.layers.sum(list(loss.values()))
            loss.update({'loss': total_loss})
            return loss
        else:
            pred = self.bbox_head.get_prediction(
                im_info, roi_feat_list, rcnn_pred_list, proposal_list,
                self.cascade_bbox_reg_weights, self.cls_agnostic_bbox_reg)
            return pred

    def _decode_box(self, proposals, bbox_pred, bbox_reg_weights):
        rcnn_loc_delta_r = fluid.layers.reshape(
            bbox_pred, (-1, self.cls_agnostic_bbox_reg, 4))
        # only use fg box delta to decode box
        rcnn_loc_delta_s = fluid.layers.slice(
            rcnn_loc_delta_r, axes=[1], starts=[1], ends=[2])
        refined_bbox = fluid.layers.box_coder(
            prior_box=proposals,
            prior_box_var=bbox_reg_weights,
            target_box=rcnn_loc_delta_s,
            code_type='decode_center_size',
            box_normalized=False,
            axis=1, )
        refined_bbox = fluid.layers.reshape(refined_bbox, shape=[-1, 4])

        return refined_bbox

    def train(self):
        """
	Run train process
	"""
        return self._forward()

    def test(self):
        """
	Run infer process
	"""
        return self._forward()
