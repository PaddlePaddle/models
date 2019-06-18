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
from ppdet.models.tests.decorator_helper import prog_scope
from ppdet.core.config import load_cfg, merge_cfg
from ppdet.models.anchor_heads.fpn_rpn_head import FPNRPNHead

import paddle.fluid as fluid

YAML_LIST = [
    'faster-rcnn_ResNet50-FPN_1x.yml',
    'faster-rcnn_ResNet50-FPN_2x.yml',
]


def init_input(cfg):
    im_info = fluid.layers.data(name='im_info', shape=[3], dtype='float32')
    fpn_2 = fluid.layers.data(
        name='fpn_res2c_sum', shape=[256, 334, 334], dtype='float32')
    fpn_3 = fluid.layers.data(
        name='fpn_res3d_sum', shape=[256, 167, 167], dtype='float32')
    fpn_4 = fluid.layers.data(
        name='fpn_res4f_sum', shape=[256, 84, 84], dtype='float32')
    fpn_5 = fluid.layers.data(
        name='fpn_res5c_sum', shape=[256, 42, 42], dtype='float32')
    fpn_6 = fluid.layers.data(
        name='fpn_res5c_sum_subsampled_2x',
        shape=[256, 21, 21],
        dtype='float32')
    fpn_list = [fpn_2, fpn_3, fpn_4, fpn_5, fpn_6]
    fpn_dict = {}
    fpn_name_list = []
    for var in fpn_list:
        fpn_dict[var.name] = var
        fpn_name_list.append(var.name)
    return fpn_dict, im_info, fpn_name_list


@prog_scope()
def test_fpn_rpn_head(cfg_file, is_train):
    cfg = load_cfg(cfg_file)
    merge_cfg({'IS_TRAIN': is_train}, cfg)
    ob = FPNRPNHead(cfg)
    input = init_input(cfg)
    rois_collect = ob.get_proposals(*input)
    im_info = input[1]
    gt_box = fluid.layers.data(
        name='gt_box', shape=[4], dtype='float32', lod_level=1)
    is_crowd = fluid.layers.data(
        name='is_crowd', shape=[1], dtype='int32', lod_level=1)
    loss_rpn_fpn_dict = ob.get_loss(im_info, gt_box, is_crowd)

    assert rois_collect is not None
    assert loss_rpn_fpn_dict is not None


class TestFPNRPNHead(unittest.TestCase):
    def test_fpn_rpn_heads_test(self):
        path = os.path.dirname(configs.__file__)
        for yml_file in YAML_LIST:
            test_fpn_rpn_head(os.path.join(path, yml_file), False)

    def test_fpn_rpn_heads_train(self):
        path = os.path.dirname(configs.__file__)
        for yml_file in YAML_LIST:
            test_fpn_rpn_head(os.path.join(path, yml_file), True)


if __name__ == "__main__":
    unittest.main()
