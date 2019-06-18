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
from ppdet.core.config import load_cfg, merge_cfg
from ppdet.models.anchor_heads.rpn_head import RPNHead
import paddle.fluid as fluid
from paddle.fluid.framework import Program, program_guard

YAML_LIST = [
    'faster-rcnn_ResNet50-C4_1x.yml',
    'faster-rcnn_ResNet50-C4_2x.yml',
    'mask-rcnn_ResNet50-C4_1x.yml',
    'mask-rcnn_ResNet50-C4_2x.yml',
]


def test_rpn_head(cfg_file, is_train):
    cfg = load_cfg(cfg_file)
    merge_cfg({'IS_TRAIN': is_train}, cfg)
    program = Program()
    with program_guard(program):
        ob = RPNHead(cfg)
        gt_box = fluid.layers.data(
            name='gt_box', shape=[4], dtype='float32', lod_level=1)
        is_crowd = fluid.layers.data(
            name='is_crowd', shape=[1], dtype='int32', lod_level=1)
        rpn_input = fluid.layers.data(
            name='rpn_input', shape=[1024, 84, 84], dtype='float32')
        name_list = ['res4_sum']
        body_dict = {name_list[0]: rpn_input}
        im_info = fluid.layers.data(name='im_info', shape=[3], dtype='float32')
        rpn_rois = ob.get_proposals(body_dict, im_info, name_list)
        rpn_loss = ob.get_loss(im_info, gt_box, is_crowd)
        rpn_cls_loss = rpn_loss['loss_rpn_cls']
        rpn_bbox_loss = rpn_loss['loss_rpn_bbox']

        assert rpn_rois is not None
        assert rpn_cls_loss is not None
        assert rpn_bbox_loss is not None


class TestRPNHead(unittest.TestCase):
    def test_rpn_head_train(self):
        path = os.path.dirname(configs.__file__)
        for yml_file in YAML_LIST:
            test_rpn_head(os.path.join(path, yml_file), True)

    def test_rpn_head_test(self):
        path = os.path.dirname(configs.__file__)
        for yml_file in YAML_LIST:
            test_rpn_head(os.path.join(path, yml_file), False)


if __name__ == "__main__":
    unittest.main()
