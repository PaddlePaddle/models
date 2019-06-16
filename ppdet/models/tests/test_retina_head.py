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
from ppdet.models.anchor_heads.retina_head import RETINAHead

import paddle.fluid as fluid

YAML_LIST = ['retinanet_ResNet50-FPN_1x.yml', ]


def init_input(cfg):
    im_info = fluid.layers.data(name='im_info', shape=[3], dtype='float32')
    fpn_3 = fluid.layers.data(
        name='fpn_res3d_sum', shape=[256, 334, 334], dtype='float32')
    fpn_4 = fluid.layers.data(
        name='fpn_res4f_sum', shape=[256, 167, 167], dtype='float32')
    fpn_5 = fluid.layers.data(
        name='fpn_res5c_sum', shape=[256, 84, 84], dtype='float32')
    fpn_6 = fluid.layers.data(
        name='fpn_6', shape=[256, 42, 42], dtype='float32')
    fpn_7 = fluid.layers.data(
        name='fpn_7', shape=[256, 21, 21], dtype='float32')
    fpn_list = [fpn_7, fpn_6, fpn_5, fpn_4, fpn_3]
    fpn_dict = {}
    fpn_name_list = []
    for var in fpn_list:
        fpn_dict[var.name] = var
        fpn_name_list.append(var.name)
    spatial_scale = [1. / 128., 1. / 64., 1. / 32., 1. / 16., 1. / 8.]
    return fpn_dict, im_info, fpn_name_list, spatial_scale


@prog_scope()
def test_retina_head(cfg_file, is_train):
    cfg = load_cfg(cfg_file)
    merge_cfg({'IS_TRAIN': is_train}, cfg)
    ob = RETINAHead(cfg)
    input = init_input(cfg)
    im_info = input[1]
    body_feats = input[0]
    spatial_scale = input[3]
    fpn_name_list = input[2]
    gt_box = fluid.layers.data(
        name='gt_box', shape=[4], dtype='float32', lod_level=1)
    gt_label = fluid.layers.data(
        name='gt_label', shape=[4], dtype='int32', lod_level=1)
    is_crowd = fluid.layers.data(
        name='is_crowd', shape=[1], dtype='int32', lod_level=1)
    if cfg.IS_TRAIN:
        loss_dict = ob.get_loss(body_feats, spatial_scale, fpn_name_list,
                                im_info, gt_box, gt_label, is_crowd)
        loss_cls = loss_dict['loss_cls']
        loss_bbox = loss_dict['loss_bbox']
        assert loss_cls is not None
        assert loss_bbox is not None
    else:
        pred_result = ob.get_prediction(body_feats, spatial_scale,
                                        fpn_name_list, im_info)
        pred_result = pred_result['bbox']
        assert pred_result is not None


class TestRETINAHead(unittest.TestCase):
    def test_retina_heads_test(self):
        path = os.path.dirname(configs.__file__)
        for yml_file in YAML_LIST:
            test_retina_head(os.path.join(path, yml_file), False)

    def test_retina_heads_train(self):
        path = os.path.dirname(configs.__file__)
        for yml_file in YAML_LIST:
            test_retina_head(os.path.join(path, yml_file), True)


if __name__ == "__main__":
    unittest.main()
