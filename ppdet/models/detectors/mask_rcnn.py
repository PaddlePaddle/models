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

import math
import six

import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.framework import Variable

from ..registry import Detectors
from ..registry import Backbones
from ..registry import RPNHeads
from ..registry import RoIExtractors
from ..registry import BBoxHeads
from ..registry import BBoxHeadConvs
from ..registry import MaskHeads

from ..target_assigners.bbox_assigner import BBoxAssigner
from ..target_assigners.mask_assigner import MaskAssigner

from .faster_rcnn import FasterRCNN

__all__ = ['MaskRCNN']


@Detectors.register
class MaskRCNN(FasterRCNN):
    """
    TODO(qingqing): add more comments
    """

    def __init__(self, cfg):
        super(MaskRCNN, self).__init__(cfg)
        self.mask_head = MaskHeads.get(cfg.MASK_HEAD.TYPE)(cfg)
        self.mask_assigner = MaskAssigner(cfg)

    def train(self):
        # inputs
        feed_vars = self.build_feeds(self.feed_info(), self.use_pyreader)
        im = feed_vars['image']
        im_info = feed_vars['im_info']
        gt_box = feed_vars['gt_box']
        is_crowd = feed_vars['is_crowd']

        # backbone
        body_feats = self.backbone(im)
        body_feat_names = self.backbone.get_body_feat_names()

        # rpn proposals
        rois = self.rpn_head.get_proposals(body_feats, im_info, body_feat_names)

        rpn_loss = self.rpn_head.get_loss(im_info, gt_box, is_crowd)

        # sampled rpn proposals
        outs = self.bbox_assigner.get_sampled_rois_and_targets(rois, feed_vars)
        rois = outs[0]
        labels_int32 = outs[1]
        bbox_targets = outs[2]
        bbox_inside_weights = outs[3]
        bbox_outside_weights = outs[4]

        # RoI Extractor
        body_feat = body_feats[body_feat_names[-1]]
        roi_feat = self.roi_extractor.get_roi_feat(body_feat, rois)

        # fast-rcnn head and rcnn loss
        loss = self.bbox_head.get_loss(roi_feat, labels_int32, bbox_targets,
                                       bbox_inside_weights,
                                       bbox_outside_weights)
        loss.update(rpn_loss)

        # mask head and mask loss
        outs = self.mask_assigner.get_mask_rois_and_targets(rois, labels_int32,
                                                            feed_vars)
        mask_rois, roi_has_mask_int32, mask_int32 = outs
        bbox_head_feat = self.bbox_head.get_head_feat()

        feat = fluid.layers.gather(bbox_head_feat, roi_has_mask_int32)
        mask_loss = self.mask_head.get_loss(feat, mask_int32)
        loss.update(mask_loss)

        total_loss = fluid.layers.sum(list(loss.values()))
        loss.update({'loss': total_loss})
        return loss

    def test(self):
        # inputs
        feed_vars = self.build_feeds(self.feed_info(), self.use_pyreader)
        im = feed_vars['image']
        im_info = feed_vars['im_info']

        # backbone
        body_feats = self.backbone(im)
        body_feat_names = self.backbone.get_body_feat_names()

        # rpn proposals
        rois = self.rpn_head.get_proposals(body_feats, im_info, body_feat_names)
        # RoI Extractor
        body_feat = body_feats[body_feat_names[-1]]
        roi_feat = self.roi_extractor.get_roi_feat(body_feat, rois)

        # bbox prediction
        bbox_pred = self.bbox_head.get_prediction(roi_feat, rois, im_info)
        bbox_pred = bbox_pred['bbox']

        # mask prediction
        head_func = BBoxHeadConvs.get(self.cfg.BBOX_HEAD.HEAD_CONV)(self.cfg)
        bbox_shape = fluid.layers.shape(bbox_pred)
        bbox_size = fluid.layers.reduce_prod(bbox_shape)
        bbox_size = fluid.layers.reshape(bbox_size, [1, 1])
        size = fluid.layers.fill_constant([1, 1], value=6, dtype='int32')
        cond = fluid.layers.less_than(x=bbox_size, y=size)

        mask_pred = fluid.layers.create_global_var(
            shape=[1], value=0.0, dtype='float32', persistable=True)

        with fluid.layers.control_flow.Switch() as switch:
            with switch.case(cond):
                fluid.layers.assign(input=bbox_pred, output=mask_pred)
            with switch.default():
                bbox = fluid.layers.slice(bbox_pred, [1], starts=[2], ends=[6])

                im_scale = fluid.layers.slice(
                    im_info, [1], starts=[2], ends=[3])
                im_scale = fluid.layers.sequence_expand(im_scale, bbox)

                mask_rois = bbox * im_scale
                mask_feat = self.roi_extractor.get_roi_feat(body_feat,
                                                            mask_rois)

                mask_feat = head_func(mask_feat)

                mask_out = self.mask_head.get_prediction(mask_feat, bbox)
                fluid.layers.assign(input=mask_out, output=mask_pred)

        return {'bbox': bbox_pred, 'mask': mask_pred}

    def feed_info(self):
        feed_info = super(MaskRCNN, self).feed_info()
        # yapf: disable
        if self.is_train:
            anno_info = [
                {'name': 'gt_mask', 'shape': [2], 'dtype': 'float32', 'lod_level': 3},
            ]
            feed_info += anno_info
        # yapf: enable
        return feed_info
