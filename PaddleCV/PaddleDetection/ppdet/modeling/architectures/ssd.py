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

from ppdet.core.workspace import register
from ppdet.modeling.ops import SSDOutputDecoder, SSDMetric

__all__ = ['SSD']


@register
class SSD(object):
    """
    Single Shot MultiBox Detector, see https://arxiv.org/abs/1512.02325

    Args:
        backbone (object): backbone instance
        multi_box_head (object): `MultiBoxHead` instance
        output_decoder (object): `SSDOutputDecoder` instance
        metric (object): `SSDMetric` instance for training
        num_classes (int): number of output classes
    """

    __category__ = 'architecture'
    __inject__ = ['backbone', 'multi_box_head', 'output_decoder', 'metric']

    def __init__(self,
                 backbone,
                 multi_box_head='MultiBoxHead',
                 output_decoder=SSDOutputDecoder().__dict__,
                 metric=SSDMetric().__dict__,
                 num_classes=21):
        super(SSD, self).__init__()
        self.backbone = backbone
        self.multi_box_head = multi_box_head
        self.num_classes = num_classes
        self.output_decoder = output_decoder
        self.metric = metric
        if isinstance(output_decoder, dict):
            self.output_decoder = SSDOutputDecoder(**output_decoder)
        if isinstance(metric, dict):
            self.metric = SSDMetric(**metric)
        
    def build(self, feed_vars, mode='train'):
        im = feed_vars['image']
        if mode == 'train' or mode == 'eval':
            gt_box = feed_vars['gt_box']
            gt_label = feed_vars['gt_label']
            difficult = feed_vars['is_difficult']

        body_feats = self.backbone(im)
        locs, confs, box, box_var = self.multi_box_head(
            inputs=body_feats, image=im, num_classes=self.num_classes)

        if mode == 'train':
            loss = fluid.layers.ssd_loss(locs, confs, gt_box, gt_label, box,
                                         box_var)
            loss = fluid.layers.reduce_sum(loss)
            return {'loss': loss}
        else:
            pred = self.output_decoder(locs, confs, box, box_var)
            if mode == 'eval':
                map_eval = self.metric(
                    pred,
                    gt_label,
                    gt_box,
                    difficult,
                    class_num=self.num_classes)
                _, accum_map = map_eval.get_map_var()
                return {'map': map_eval, 'accum_map': accum_map}
            else:
                return {'bbox': pred}

    def train(self, feed_vars):
        return self.build(feed_vars, 'train')

    def eval(self, feed_vars):
        return self.build(feed_vars, 'eval')

    def test(self, feed_vars):
        return self.build(feed_vars, 'test')

    def is_bbox_normalized(self):
        # SSD use output_decoder in output layers, bbox is normalized
        # to range [0, 1], is_bbox_normalized is used in infer.py
        return True

