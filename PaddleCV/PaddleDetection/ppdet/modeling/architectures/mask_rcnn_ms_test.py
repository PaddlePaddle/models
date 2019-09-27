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

__all__ = ['Mask_sub_MSTEST']


@register
class Mask_sub_MSTEST(object):
    """
    Mask R-CNN architecture, see https://arxiv.org/abs/1703.06870
    Args:
        backbone (object): backbone instance
        roi_extractor (object): ROI extractor instance
        bbox_head (object): `BBoxHead` instance
        mask_head (object): `MaskHead` instance
        fpn (object): feature pyramid network instance
    """

    __category__ = 'architecture'
    __inject__ = [
        'backbone', 'roi_extractor', 'bbox_head', 'rpn_head', 'mask_head', 'fpn'
    ]

    def __init__(self,
                 backbone,
                 rpn_head,
                 bbox_head='BBoxHead',
                 roi_extractor='RoIAlign',
                 mask_head='MaskHead',
                 fpn=None):
        super(Mask_sub_MSTEST, self).__init__()
        self.backbone = backbone
        self.rpn_head = rpn_head
        self.roi_extractor = roi_extractor
        self.bbox_head = bbox_head
        self.mask_head = mask_head
        self.fpn = fpn

    def build(self, feed_vars):
        required_fields = ['im_info', 'bbox']
        for var in required_fields:
            assert var in feed_vars, \
                "{} has no {} field".format(feed_vars, var)

        ims = []
        for k in feed_vars.keys():
            if 'image' in k:
                ims.append(feed_vars[k])

        result = {}
        for i, im in enumerate(ims):
            mask_name = 'mask_pred_' + str(i)
            bbox_pred = feed_vars['bbox']
            result.update({im.name: im})
            if 'flip' in im.name:
                mask_name += '_flip'
                bbox_pred = feed_vars['bbox_flip']

            im_info = fluid.layers.slice(
                input=feed_vars['im_info'],
                axes=[1],
                starts=[3 * i],
                ends=[3 * i + 3])

            body_feats = self.backbone(im)
            result.update(body_feats)
            # FPN
            if self.fpn is not None:
                body_feats, spatial_scale = self.fpn.get_output(body_feats)
            rois = self.rpn_head.get_proposals(body_feats, im_info, mode='test')
            if self.fpn is not None:
                roi_feat = self.roi_extractor(body_feats, rois, spatial_scale)
            else:
                last_feat = body_feats[list(body_feats.keys())[-1]]
                roi_feat = self.roi_extractor(last_feat, rois)

            # share weight
            bbox_shape = fluid.layers.shape(bbox_pred)
            bbox_size = fluid.layers.reduce_prod(bbox_shape)
            bbox_size = fluid.layers.reshape(bbox_size, [1, 1])
            size = fluid.layers.fill_constant([1, 1], value=6, dtype='int32')
            cond = fluid.layers.less_than(x=bbox_size, y=size)

            mask_pred = fluid.layers.create_global_var(
                shape=[1],
                value=0.0,
                dtype='float32',
                persistable=False,
                name=mask_name)

            with fluid.layers.control_flow.Switch() as switch:
                with switch.case(cond):
                    fluid.layers.assign(input=bbox_pred, output=mask_pred)
                with switch.default():
                    bbox = fluid.layers.slice(
                        bbox_pred, [1], starts=[2], ends=[6])

                    im_scale = fluid.layers.slice(
                        im_info, [1], starts=[2], ends=[3])
                    im_scale = fluid.layers.sequence_expand(im_scale, bbox)

                    mask_rois = bbox * im_scale
                    if self.fpn is None:
                        mask_feat = self.roi_extractor(last_feat, mask_rois)
                        mask_feat = self.bbox_head.get_head_feat(mask_feat)
                    else:
                        mask_feat = self.roi_extractor(
                            body_feats, mask_rois, spatial_scale, is_mask=True)

                    mask_out = self.mask_head.get_prediction(mask_feat, bbox)
                    fluid.layers.assign(input=mask_out, output=mask_pred)
            result[mask_name] = mask_pred
        return result

    def eval(self, feed_vars):
        return self.build(feed_vars)
