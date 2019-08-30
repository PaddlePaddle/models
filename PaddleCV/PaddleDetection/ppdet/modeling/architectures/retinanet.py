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

from collections import OrderedDict

import paddle.fluid as fluid

from ppdet.experimental import mixed_precision_global_state
from ppdet.core.workspace import register

__all__ = ['RetinaNet']


@register
class RetinaNet(object):
    """
    RetinaNet architecture, see https://arxiv.org/abs/1708.02002

    Args:
        backbone (object): backbone instance
        fpn (object): feature pyramid network instance
        retina_head (object): `RetinaHead` instance
    """

    __category__ = 'architecture'
    __inject__ = ['backbone', 'fpn', 'retina_head']

    def __init__(self, backbone, fpn, retina_head):
        super(RetinaNet, self).__init__()
        self.backbone = backbone
        self.fpn = fpn
        self.retina_head = retina_head

    def build(self, mode='train'):
        im = fluid.layers.data(
            name='image', shape=[3, 800, 1333], dtype='float32')
        im_info = fluid.layers.data(name='im_info', shape=[3], dtype='float32')
        if mode == 'train':
            gt_box = fluid.layers.data(
                name='gt_box', shape=[4], dtype='float32', lod_level=1)
            gt_label = fluid.layers.data(
                name='gt_label', shape=[1], dtype='int32', lod_level=1)
            is_crowd = fluid.layers.data(
                name='is_crowd', shape=[1], dtype='int32', lod_level=1)

        mixed_precision_enabled = mixed_precision_global_state() is not None
        # cast inputs to FP16
        if mixed_precision_enabled:
            im = fluid.layers.cast(im, 'float16')

        # backbone
        body_feats = self.backbone(im)

        # cast features back to FP32
        if mixed_precision_enabled:
            body_feats = OrderedDict((k, fluid.layers.cast(v, 'float32'))
                                     for k, v in body_feats.items())

        # FPN
        body_feats, spatial_scale = self.fpn.get_output(body_feats)

        # retinanet head
        if mode == 'train':
            loss = self.retina_head.get_loss(body_feats, spatial_scale, im_info,
                                             gt_box, gt_label, is_crowd)
            total_loss = fluid.layers.sum(list(loss.values()))
            loss.update({'loss': total_loss})
            return loss
        else:
            pred = self.retina_head.get_prediction(body_feats, spatial_scale,
                                                   im_info)
            return pred

    def train(self):
        return self.build('train')

    def eval(self):
        return self.build('test')

    def test(self):
        return self.build('test')
