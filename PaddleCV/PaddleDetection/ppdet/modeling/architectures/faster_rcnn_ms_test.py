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

__all__ = ['FasterRCNN_MSTEST']


@register
class FasterRCNN_MSTEST(object):
    """
    Faster R-CNN architecture, see https://arxiv.org/abs/1506.01497
    Args:
        backbone (object): backbone instance
        rpn_head (object): `RPNhead` instance
        roi_extractor (object): ROI extractor instance
        bbox_head (object): `BBoxHead` instance
        fpn (object): feature pyramid network instance
    """

    __category__ = 'architecture'
    __inject__ = ['backbone', 'rpn_head', 'roi_extractor', 'bbox_head', 'fpn']

    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_extractor,
                 bbox_head='BBoxHead',
                 fpn=None):
        super(FasterRCNN_MSTEST, self).__init__()
        self.backbone = backbone
        self.rpn_head = rpn_head
        self.roi_extractor = roi_extractor
        self.bbox_head = bbox_head
        self.fpn = fpn

    def build(self, feed_vars):
        ims = []
        for k in feed_vars.keys():
            if 'image' in k:
                ims.append(feed_vars[k])
        result = {}
        result.update(feed_vars)
        for i, im in enumerate(ims):
            im_info = fluid.layers.slice(
                input=feed_vars['im_info'],
                axes=[1],
                starts=[3 * i],
                ends=[3 * i + 3])
            im_shape = feed_vars['im_shape']
            body_feats = self.backbone(im)
            result.update(body_feats)
            body_feat_names = list(body_feats.keys())

            if self.fpn is not None:
                body_feats, spatial_scale = self.fpn.get_output(body_feats)

            rois = self.rpn_head.get_proposals(body_feats, im_info, mode='test')

            if self.fpn is None:
                # in models without FPN, roi extractor only uses the last level of
                # feature maps. And body_feat_names[-1] represents the name of
                # last feature map.
                body_feat = body_feats[body_feat_names[-1]]
                roi_feat = self.roi_extractor(body_feat, rois)
            else:
                roi_feat = self.roi_extractor(body_feats, rois, spatial_scale)

            pred = self.bbox_head.get_prediction(
                roi_feat, rois, im_info, im_shape, return_box_score=True)
            bbox_name = 'bbox_' + str(i)
            score_name = 'score_' + str(i)
            if 'flip' in im.name:
                bbox_name += '_flip'
                score_name += '_flip'
            result[bbox_name] = pred['bbox']
            result[score_name] = pred['score']
        return result

    def eval(self, feed_vars):
        return self.build(feed_vars)
