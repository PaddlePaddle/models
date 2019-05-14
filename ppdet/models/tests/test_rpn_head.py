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
from ppdet.models.anchor_heads.rpn_heads import RPNHead
from ppdet.core.config import load_cfg
import configs
import paddle.fluid as fluid
from paddle.fluid.framework import Program, program_guard
import os

YAML_LIST = [
    'faster-rcnn_ResNet50-C4_1x.yml',
    'faster-rcnn_ResNet50-C4_2x.yml',
    'mask-rcnn_ResNet50-C4_1x.yml',
    'mask-rcnn_ResNet50-C4_2x.yml',
]


def init_input(cfg):
    out = [cfg]
    im_info = fluid.layers.data(name='im_info', shape=[3], dtype='float32')
    out.append(im_info)
    return out


def test_rpn_head(cfg_file):
    cfg = load_cfg(cfg_file)
    program = Program()
    with program_guard(program):
        out = init_input(cfg)
        ob = RPNHead(*out)
        gt_box = fluid.layers.data(
            name='gt_box', shape=[4], dtype='float32', lod_level=1)
        is_crowd = fluid.layers.data(
            name='is_crowd', shape=[1], dtype='int32', lod_level=1)
        rpn_input = fluid.layers.data(
            name='rpn_input', shape=[1024, 84, 84], dtype='float32')
        rpn_cls_score, rpn_bbox_pred = ob.get_output(rpn_input)
        rpn_rois, rpn_roi_probs = ob.get_proposals(rpn_cls_score, rpn_bbox_pred)
        rpn_cls_loss, rpn_bbox_loss = ob.get_loss(rpn_cls_score, rpn_bbox_pred,
                                                  gt_box, is_crowd)

        assert rpn_cls_score is not None
        assert rpn_bbox_pred is not None
        assert rpn_rois is not None
        assert rpn_roi_probs is not None
        assert rpn_cls_loss is not None
        assert rpn_bbox_loss is not None


class TestRPNHead(unittest.TestCase):
    def test_rpn_head(self):
        path = os.path.dirname(configs.__file__)
        for yml_file in YAML_LIST:
            test_rpn_head(os.path.join(path, yml_file))


if __name__ == "__main__":
    unittest.main()
