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
import models.roi_extractors as roi_extractors
from core.config import load_cfg
import paddle.fluid as fluid
from paddle.fluid.framework import Program, program_guard

YAML_LIST = [
    'faster-rcnn_ResNet50-C4_1x.yml',
    'faster-rcnn_ResNet50-C4_2x.yml',
    'mask-rcnn_ResNet50-C4_1x.yml',
    'mask-rcnn_ResNet50-C4_2x.yml',
    'faster-rcnn_ResNet50-FPN_1x.yml',
    'faster-rcnn_ResNet50-FPN_2x.yml',
]


def init_head_input(cfg):
    if cfg.MODEL.FPN:
        head_inputs = [
            fluid.layers.data(
                name='head_inputs_0',
                shape=[256, 21, 21],
                dtype='float32',
                lod_level=1),
            fluid.layers.data(
                name='head_inputs_1',
                shape=[256, 42, 42],
                dtype='float32',
                lod_level=1),
            fluid.layers.data(
                name='head_inputs_2',
                shape=[256, 84, 84],
                dtype='float32',
                lod_level=1),
            fluid.layers.data(
                name='head_inputs_3',
                shape=[256, 334, 334],
                dtype='float32',
                lod_level=1),
        ]

    else:
        head_inputs = fluid.layers.data(
            name='head_inputs',
            shape=[1024, 84, 84],
            dtype='float32',
            lod_level=1)
    return head_inputs


def test_roi_extractor(cfg_file):
    cfg = load_cfg(cfg_file)
    program = Program()
    with program_guard(program):
        method = getattr(roi_extractors, cfg.ROI_EXTRACTOR.EXTRACT_METHOD)
        ob = method(cfg)
        head_inputs = init_head_input(cfg)
        rois = fluid.layers.data(
            name='rois', shape=[4], dtype='int32', lod_level=1)
        roi_feat = ob.get_roi_feat(head_inputs, rois)

        assert roi_feat is not None


class TestRPNHead(unittest.TestCase):
    def test_rpn_head(self):
        for yml_file in YAML_LIST:
            test_roi_extractor('../configs/' + yml_file)


if __name__ == "__main__":
    unittest.main()
