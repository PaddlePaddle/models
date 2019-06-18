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

from ppdet.core.workspace import register, serializable

__all__ = ['OutputDecoder', 'SSDMetric', 'SSD']


@register
@serializable
class OutputDecoder(object):
    __op__ = fluid.layers.detection_output
    __append_doc__ = True

    def __init__(self,
                 nms_threshold=0.45,
                 nms_top_k=400,
                 keep_top_k=200,
                 score_threshold=0.01,
                 nms_eta=1.0,
                 background_label=0):
        super(OutputDecoder, self).__init__()
        self.nms_threshold = nms_threshold
        self.background_label = background_label
        self.nms_top_k = nms_top_k
        self.keep_top_k = keep_top_k
        self.score_threshold = score_threshold
        self.nms_eta = nms_eta


@register
@serializable
class SSDMetric(object):
    __op__ = fluid.metrics.DetectionMAP
    __append_doc__ = True

    def __init__(self,
                 overlap_threshold=0.5,
                 evaluate_difficult=False,
                 ap_version='integral'):
        super(SSDMetric, self).__init__()
        self.overlap_threshold = overlap_threshold
        self.evaluate_difficult = evaluate_difficult
        self.ap_version = ap_version


@register
class SSD(object):
    r"""
    Single Shot MultiBox Detector, see https://arxiv.org/abs/1512.02325

    Args:
        backbone (object): backbone instance
        multi_box_head (object): `MultiBoxHead` instance
        output_decoder (object): `OutputDecoder` instance
        metric (object): `SSDMetric` instance for training
        num_classes (int): number of output classes
    """

    __category__ = 'architecture'
    __inject__ = ['backbone', 'multi_box_head', 'output_decoder', 'metric']

    def __init__(self,
                 backbone,
                 multi_box_head='MultiBoxHead',
                 output_decoder=OutputDecoder().__dict__,
                 metric=SSDMetric().__dict__,
                 num_classes=21):
        super(SSD, self).__init__()
        self.backbone = backbone
        self.multi_box_head = multi_box_head
        self.num_classes = num_classes
        self.output_decoder = output_decoder
        self.metric = metric
        if isinstance(output_decoder, dict):
            self.output_decoder = OutputDecoder(**output_decoder)
        if isinstance(metric, dict):
            self.metric = SSDMetric(**metric)

    def _forward(self, feed_vars, mode='train'):
        im = feed_vars['image']
        if mode == 'train' or mode == 'eval':
            gt_box = feed_vars['gt_box']
            gt_label = feed_vars['gt_label']
            difficult = feed_vars['is_difficult']

        body_feats = self.backbone(im)
        locs, confs, box, box_var = self.multi_box_head(
            inputs=body_feats, image=im, num_classes=self.num_classes)

        if mode == 'train':
            loss = fluid.layers.ssd_loss(
                locs, confs, gt_box, gt_label, box, box_var)
            loss = fluid.layers.reduce_sum(loss)
            return {'loss': loss}
        else:
            pred = self.output_decoder(locs, confs, box, box_var)
            if mode == 'eval':
                map_eval = self.metric(pred, gt_box, gt_label, difficult,
                                       class_num=self.num_classes)
                return {'map': map_eval}
            else:
                return {'bbox': pred}

    def train(self, feed_vars):
        return self._forward(feed_vars, 'train')

    def eval(self, feed_vars):
        return self._forward(feed_vars, 'eval')

    def test(self, feed_vars):
        return self._forward('test')
