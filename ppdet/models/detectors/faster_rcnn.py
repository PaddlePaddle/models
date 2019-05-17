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
from ..registry import RPNHead
from ..registry import RoIExtractor
from ..registry import BBoxHeads

from ..target_assigners.bbox_assigner import BBoxAssigner


@Detectors.register
class FasterRCNN(DetectorBase):
    def __init__(self, cfg):
        super(FasterRCNN, self).__init__(cfg)
        self.is_train = cfg.IS_TRAIN
        self.backbone = Backbones.get(cfg.MODEL.BACKBONE)(cfg)
        self.rpn_head = RPNHead.get(cfg.RPN_HEAD.TYPE)(cfg)
        self.bbox_assigner = BBoxAssigner(cfg)
        self.roi_extractor = RoIExtractor.get(
            [cfg.ROI_EXTRACTOR.EXTRACT_METHOD])(cfg)
        self.bbox_head = BBoxHeads.get([cfg.BBOX_HEAD.HEAD_CONV])(cfg)
        self.bbox_head_conv = BBoxHeads.get([cfg.BBOX_HEAD.HEAD_CONV])(cfg)

    def train(self):
        # inputs
        feed_vars = self.build_feeds(self.feed_info())
        im = feed_vars['image']
        im_info = feed_vars['im_info']
        gt_box = feed_vars['gt_box']
        is_crowd = feed_vars['is_crowd']

        # backbone
        body_feat = self.backbone(im)

        # rpn proposals
        rpn_rois, rpn_roi_probs = self.rpn_head.get_proposals(body_feat)
        self.rpn_head.get_loss(im_info, gt_box, is_crowd)

        # sampled rpn proposals
        outs = self.bbox_assigner.get_sampled_rois_and_targets(rpn_rois,
                                                               feed_vars)
        rois, labels_int32, bbox_targets, bbox_inside_weights, bbox_outside_weights = outs

        # RoI Extractor
        roi_feat = self.roi_extractor.get_roi_feat(body_feat, rois)

        # fast-rcnn head

        #def test(self):

    def feed_info(self):
        c = getattr(cfg.DATA.IM_CHANNEL, 3)
        h = getattr(cfg.DATA.IM_HEIGHT, 224)
        w = getattr(cfg.DATA.IM_WIDTH, 224)
        feed_info = [
            {
                'name': 'image',
                'shape': [c, h, w],
                'dtype': 'float32',
                'lod_level': 0
            },
            {
                'name': 'im_info',
                'shape': [1],
                'dtype': 'float32',
                'lod_level': 0
            },
        ]
        if self.is_train:
            anno_info = [
                {
                    'name': 'gt_box',
                    'shape': [1],
                    'dtype': 'float32',
                    'lod_level': 1
                },
                {
                    'name': 'gt_label',
                    'shape': [1],
                    'dtype': 'int32',
                    'lod_level': 1
                },
                {
                    'name': 'is_crowd',
                    'shape': [1],
                    'dtype': 'int32',
                    'lod_level': 1
                },
                {
                    'name': 'im_id',
                    'shape': [1],
                    'dtype': 'int32',
                    'lod_level': 0
                },
            ]
            feed_info += anno_info
        return feed_info
