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
from __future__ import unicode_literals

import unittest
from models.anchor_heads.rpn_heads import RPNHead
from core.config import load_cfg
import paddle.fluid as fluid
from paddle.fluid.framework import Program, program_guard

YAML_LIST = [
    'faster-rcnn_ResNet50-C4_1x.yml',
    'faster-rcnn_ResNet50-C4_2x.yml',
    'mask-rcnn_ResNet50-C4_1x.yml',
    'mask-rcnn_ResNet50-C4_2x.yml',
]


def init_input(cfg, mask_on):
    out = [cfg]
    gt_box = fluid.layers.data(
        name='gt_box', shape=[4], dtype='float32', lod_level=1)
    gt_label = fluid.layers.data(
        name='gt_label', shape=[1], dtype='int32', lod_level=1)
    is_crowd = fluid.layers.data(
        name='is_crowd', shape=[1], dtype='int32', lod_level=1)
    im_info = fluid.layers.data(name='im_info', shape=[3], dtype='float32')
    out.append(gt_box)
    out.append(gt_label)
    out.append(is_crowd)
    out.append(im_info)
    if mask_on:
        gt_masks = fluid.layers.data(
            name='gt_masks', shape=[2], dtype='float32', lod_level=3)
        out.append(gt_masks)
    return out


def test_rpn_head(cfg_file, mask_on):
    cfg = load_cfg(cfg_file)
    program = Program()
    with program_guard(program):
        out = init_input(cfg, mask_on)
        ob = RPNHead(*out)
        rpn_input = fluid.layers.data(
            name='rpn_input', shape=[1024, 84, 84], dtype='float32')
        anchor, var, rpn_cls_score, rpn_bbox_pred = ob.get_output(rpn_input)
        rpn_rois, rpn_roi_probs = ob.get_poposals(rpn_cls_score, rpn_bbox_pred,
                                                  anchor, var)
        rois, labels_int32, bbox_targets, bbox_inside_weights, bbox_outside_weights = ob.get_proposal_targets(
            rpn_rois)
        rpn_cls_loss, rpn_bbox_loss = ob.get_loss(rpn_cls_score, rpn_bbox_pred,
                                                  anchor, var)
        if mask_on:
            mask_rois, roi_has_mask_int32, mask_int32 = ob.get_mask_targets(
                rois, labels_int32)
            assert mask_rois is not None
            assert roi_has_mask_int32 is not None
            assert mask_int32 is not None

        assert anchor is not None
        assert var is not None
        assert rpn_cls_score is not None
        assert rpn_bbox_pred is not None
        assert rois is not None
        assert labels_int32 is not None
        assert bbox_targets is not None
        assert bbox_inside_weights is not None
        assert bbox_inside_weights is not None
        assert rpn_cls_loss is not None
        assert rpn_bbox_loss is not None


class TestRPNHead(unittest.TestCase):
    def test_rpn_head(self):
        for yml_file in YAML_LIST:
            mask_on = 'mask' in yml_file
            test_rpn_head('../configs/' + yml_file, mask_on)


if __name__ == "__main__":
    unittest.main()
