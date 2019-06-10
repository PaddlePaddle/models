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
from ..registry import SSDHeads
from .base import DetectorBase
__all__ = ['SSD']


@Detectors.register
class SSD(DetectorBase):
    def __init__(self, cfg):
        super(SSD, self).__init__(cfg)
        self.mode = cfg.MODE
        self.backbone = Backbones.get(cfg.MODEL.BACKBONE)(cfg, True)
        self.ssd_head = SSDHeads.get(cfg.SSD_HEAD.TYPE)(cfg)
        self.use_pyreader = True if self.mode == 'train' or self.mode == 'val' else False

    def _forward(self):
        # inputs
        feed_vars = self.build_feeds(self.feed_info(), self.use_pyreader)
        im = feed_vars['image']
        if self.mode == 'train' or self.mode == 'val':
            gt_box = feed_vars['gt_box']
            gt_label = feed_vars['gt_label']
            is_difficult = feed_vars['is_difficult']
        # backbone
        body_feat1, body_feat2, body_feat3, body_feat4, body_feat5, body_feat6 = self.backbone(
            im)
        body_feats = [
            body_feat1, body_feat2, body_feat3, body_feat4, body_feat5,
            body_feat6
        ]
        # ssd head
        if self.mode == 'train':
            loss = self.ssd_head.get_loss(im, body_feats, gt_box, gt_label)
            return loss
        else:
            pred = self.ssd_head.get_prediction(im, body_feats)
            if self.mode == 'val':
                map_eval = self.ssd_head.get_map(gt_box, gt_label, is_difficult)
                return map_eval
            else:
                return pred

    def train(self):
        return self._forward()

    def val(self):
        return self._forward()

    def test(self):
        return self._forward()

    def feed_info(self):
        c = getattr(self.cfg.DATA, 'IM_CHANNEL', 3)
        h = getattr(self.cfg.DATA, 'TARGET_SIZE', 300)
        w = getattr(self.cfg.DATA, 'TARGET_SIZE', 300)
        # yapf: disable
        feed_info = [
            {'name': 'image',  'shape': [c, h, w], 'dtype': 'float32', 'lod_level': 0},
        ]
        if self.mode == 'train' or self.mode == 'val':
            anno_info = [
                {'name': 'gt_box',  'shape': [4], 'dtype': 'float32', 'lod_level': 1},
                {'name': 'gt_label','shape': [1], 'dtype': 'int32', 'lod_level': 1},
                {'name': 'is_difficult', 'shape': [1],'dtype': 'int32', 'lod_level': 1},
            ]
            feed_info += anno_info
        # yapf: enable
        return feed_info
