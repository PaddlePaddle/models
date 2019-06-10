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
from ..registry import YOLOHeads

from .base import DetectorBase

__all__ = ['YOLOv3']


@Detectors.register
class YOLOv3(DetectorBase):
    def __init__(self, cfg):
        super(YOLOv3, self).__init__(cfg)
        self.backbone = Backbones.get(cfg.MODEL.BACKBONE)(cfg)
        self.yolo_head = YOLOHeads.get(cfg.YOLO_HEAD.TYPE)(cfg)
        self.use_pyreader = True

    def _forward(self, is_train=False):
        self.is_train = is_train

        # build inputs
        feed_vars = self.build_feeds(
            self.feed_info(), use_pyreader=self.use_pyreader)

        # backbone
        im = feed_vars['image']
        body_feats = self.backbone(im)
        if isinstance(body_feats, dict):
            # if body_feats in a dict, get the feats list in stage order
            body_feat_names = self.backbone.get_body_feat_names()
            body_feats = [body_feats[name] for name in body_feat_names]

        if is_train:
            # get loss in train mode
            gt_box = feed_vars['gt_box']
            gt_label = feed_vars['gt_label']
            gt_score = feed_vars['gt_score']
            
            loss = {'loss': \
                    self.yolo_head.get_loss(body_feats, gt_box, gt_label, gt_score)}
            return loss
        else:
            # get prediction in test mode
            im_shape = feed_vars['im_shape']

            pred = self.yolo_head.get_prediction(body_feats, im_shape)
            return pred

    def train(self):
        return self._forward(is_train=True)

    def test(self):
        return self._forward(is_train=False)

    def feed_info(self):
        size = getattr(self.cfg.DATA, 'INPUT_SIZE', 608)
        box_num = getattr(self.cfg.DATA, 'MAX_BOX_NUM', 50)

        # yapf: disable
        feed_info = [
            {'name': 'image',  'shape': [3, size, size], 'dtype': 'float32', 'lod_level': 0},
        ]
        if self.is_train:
            train_info = [
                {'name': 'gt_box',  'shape': [box_num, 4], 'dtype': 'float32', 'lod_level': 0},
                {'name': 'gt_label','shape': [box_num], 'dtype': 'int32', 'lod_level': 0},
                {'name': 'gt_score','shape': [box_num], 'dtype': 'float32', 'lod_level': 0},
            ]
            feed_info += train_info
        else:
            test_info = [
                {'name': 'im_shape', 'shape': [2], 'dtype': 'int32', 'lod_level': 0},
                {'name': 'im_id', 'shape': [1], 'dtype': 'int32', 'lod_level': 0},
            ]
            feed_info += test_info
        # yapf: enable
        return feed_info
