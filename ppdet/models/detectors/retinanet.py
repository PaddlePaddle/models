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
#(luoqianhui): change comment stype above in github

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
from ..registry import Necks
from ..registry import RetinaHeads

from .base import DetectorBase

__all__ = ['Retinanet']


@Detectors.register
class Retinanet(DetectorBase):
    """
    Retinanet class
    
    Args:
        cfg(dict): All parameters in dictionary
    """

    def __init__(self, cfg):
        super(Retinanet, self).__init__(cfg)
        self.is_train = cfg.IS_TRAIN
        self.NECK_ON = getattr(cfg.MODEL, 'NECK', None)
        self.backbone = Backbones.get(cfg.MODEL.BACKBONE)(cfg)
        self.retina_head = RetinaHeads.get(cfg.RETINA_HEAD.TYPE)(cfg)
        if self.NECK_ON:
            self.neck = Necks.get(cfg.MODEL.NECK)(cfg)
        self.use_pyreader = True

    def _forward(self):
        # inputs
        feed_vars = self.build_feeds(self.feed_info(), self.use_pyreader)
        im = feed_vars['image']
        im_info = feed_vars['im_info']
        if self.is_train:
            gt_box = feed_vars['gt_box']
            gt_label = feed_vars['gt_label']
            is_crowd = feed_vars['is_crowd']

        # backbone
        body_feats = self.backbone(im)
        body_feat_names = self.backbone.get_body_feat_names()

        # neck
        if self.NECK_ON:
            body_feats, spatial_scale, body_feat_names = self.neck.get_output(
                body_feats, body_feat_names)

        # retinanet head
        if self.is_train:
            loss = self.retina_head.get_loss(body_feats, spatial_scale,
                                             body_feat_names, im_info, gt_box,
                                             gt_label, is_crowd)
            total_loss = fluid.layers.sum(list(loss.values()))
            loss.update({'loss': total_loss})
            return loss
        else:
            pred = self.retina_head.get_prediction(body_feats, spatial_scale,
                                                   body_feat_names, im_info)
            return pred

    def train(self):
        """
        Get the focal loss and smooth L1 loss
        """
        return self._forward()

    def test(self):
        """
        Get the class and bounding box predictions
        """
        return self._forward()

    def feed_info(self):
        """
        Set the input data needed by retinanet
        """
        c = getattr(self.cfg.DATA, 'IM_CHANNEL', 3)
        h = getattr(self.cfg.DATA, 'IM_HEIGHT', 224)
        w = getattr(self.cfg.DATA, 'IM_WIDTH', 224)

        # the order of data layers shoule be the same as
        # them ppdet/dataset/transform/operator/arrange_sample.py
        # yapf: disable
        feed_info = [
            {'name': 'image', 'shape': [c, h, w], 'dtype': 'float32', 'lod_level': 0},
            {'name': 'im_info', 'shape': [3], 'dtype': 'float32', 'lod_level': 0},
            {'name': 'im_id', 'shape': [1], 'dtype': 'int32', 'lod_level': 0},
        ]
        if self.is_train:
            anno_info = [
                {'name': 'gt_box', 'shape': [4], 'dtype': 'float32', 'lod_level': 1},
                {'name': 'gt_label', 'shape': [1], 'dtype': 'int32', 'lod_level': 1},
                {'name': 'is_crowd', 'shape': [1], 'dtype': 'int32', 'lod_level': 1},
            ]
            feed_info += anno_info
        # yapf: enable
        return feed_info
