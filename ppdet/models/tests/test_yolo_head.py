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

import os
import unittest
import configs

from ppdet.core.config import load_cfg
from ppdet.models.registry import YOLOHeads

import paddle.fluid as fluid
from paddle.fluid.framework import Program, program_guard

YAML_LIST = ['yolov3_DarkNet53_1x_syncbn.yml']


def build_input():
    gt_box = fluid.layers.data(
        name='gt_box', shape=[50, 4], dtype='float32', lod_level=0)
    gt_label = fluid.layers.data(
        name='gt_label', shape=[50], dtype='float32', lod_level=0)
    gt_score = fluid.layers.data(
        name='gt_score', shape=[50], dtype='float32', lod_level=0)

    im_shape = fluid.layers.data(
        name='im_shape', shape=[2], dtype='int32', lod_level=0)

    inputs = []
    channel = 64
    shape = 304
    for i in range(5):
        inputs.append(
            fluid.layers.data(
                name='input{}'.format(i),
                shape=[channel, shape, shape],
                dtype='float32',
                lod_level=0))
        channel *= 2
        shape //= 2

    return gt_box, gt_label, gt_score, im_shape, inputs


def test_yolo_head(cfg_file):
    cfg = load_cfg(cfg_file)
    prog = Program()
    with program_guard(prog):
        yolo_head = YOLOHeads.get(cfg.YOLO_HEAD.TYPE)(cfg)
        gt_box, gt_label, gt_score, im_shape, inputs = build_input()
        loss = yolo_head.get_loss(inputs, gt_box, gt_label, gt_score)
        pred = yolo_head.get_prediction(inputs, im_shape)

        assert loss is not None
        assert pred is not None


class TestYOLOHeads(unittest.TestCase):
    def test_yolo_head(self):
        path = os.path.dirname(configs.__file__)
        for yml_file in YAML_LIST:
            test_yolo_head(os.path.join(path, yml_file))


if __name__ == "__main__":
    unittest.main()
