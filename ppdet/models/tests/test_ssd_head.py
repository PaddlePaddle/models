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

from ppdet.core.config import load_cfg
import unittest
import configs
import paddle.fluid as fluid
from paddle.fluid.framework import Program, program_guard

from ppdet.models.anchor_heads import ssd_head
import os


def init_ssd_input(cfg):
    img = fluid.layers.data(
        name='input', shape=[3, 300, 300], dtype='float32', lod_level=0)
    fm1 = fluid.layers.data(
        name='featuremap1', shape=[512, 19, 19], dtype='float32', lod_level=0)
    fm2 = fluid.layers.data(
        name='featuremap2', shape=[1024, 10, 10], dtype='float32', lod_level=0)
    fm3 = fluid.layers.data(
        name='featuremap3', shape=[512, 10, 10], dtype='float32', lod_level=0)
    fm4 = fluid.layers.data(
        name='featuremap4', shape=[256, 5, 5], dtype='float32', lod_level=0)
    fm5 = fluid.layers.data(
        name='featuremap5', shape=[256, 3, 3], dtype='float32', lod_level=0)
    fm6 = fluid.layers.data(
        name='featuremap6', shape=[128, 2, 2], dtype='float32', lod_level=0)
    gt_label = fluid.layers.data(
        name='gt_label', shape=[1], dtype='int32', lod_level=1)
    gt_box = fluid.layers.data(
        name='gt_box', shape=[4], dtype='float32', lod_level=1)
    is_difficult = fluid.layers.data(
        name='difficult', shape=[1], dtype='float32', lod_level=1)
    return img, fm1, fm2, fm3, fm4, fm5, fm6, gt_label, gt_box, is_difficult


def test_ssd_head(cfg_file):
    cfg = load_cfg(cfg_file)
    main_program = Program()
    start_program = Program()
    with program_guard(main_program, start_program):
        ob = ssd_head.SSDHead(cfg)
        head_inputs = init_ssd_input(cfg)
        img = head_inputs[0]
        fm1 = head_inputs[1]
        fm2 = head_inputs[2]
        fm3 = head_inputs[3]
        fm4 = head_inputs[4]
        fm5 = head_inputs[5]
        fm6 = head_inputs[6]
        fms = [fm1, fm2, fm3, fm4, fm5, fm6]
        gt_label = head_inputs[7]
        gt_box = head_inputs[8]
        is_difficult = head_inputs[9]
        nmsed_out = ob.get_prediction(img, fms)
        loss = ob.get_loss(img, fms, gt_box, gt_label)
        map_eval = ob.get_map(gt_box, gt_label, is_difficult)

        assert nmsed_out is not None
        assert loss is not None
        assert map_eval is not None


class TestBBoxHead(unittest.TestCase):
    def test_bbox_heads(self):
        path = "configs/ssd_MobileNet_1x.yml"
        test_ssd_head(path)


if __name__ == '__main__':
    unittest.main()
