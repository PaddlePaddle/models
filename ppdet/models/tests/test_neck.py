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

import unittest
import configs
import paddle.fluid as fluid

from ppdet.core.config import load_cfg
from ppdet.models.tests.decorator_helper import prog_scope
import ppdet.models.necks as neck
from ppdet.models.backbones.resnet import ResNet50Backbone
import os

YAML_LIST = [
    'faster-rcnn_ResNet50-FPN_1x.yml',
    'faster-rcnn_ResNet50-FPN_2x.yml',
]


def init_head_input(cfg):
    data = fluid.layers.data(
        name='data', shape=[3, 1333, 1333], dtype='float32')
    return data


@prog_scope()
def test_neck(cfg_file):
    cfg = load_cfg(cfg_file)
    backbone = ResNet50Backbone(cfg)
    neck_ob = neck.FPN(cfg)
    data = init_head_input(cfg)
    body_feat_dict = backbone(data)
    body_name_list = backbone.get_body_feat_names()
    fpn_dict, spatial_scale, fpn_name_list = neck_ob.get_output(body_feat_dict,
                                                                body_name_list)
    assert len(spatial_scale) != 0
    assert len(spatial_scale) == len(fpn_name_list)
    for name in fpn_name_list:
        assert fpn_dict[name] is not None


class TestBBoxHead(unittest.TestCase):
    def test_necks(self):
        path = os.path.dirname(configs.__file__)
        for yml_file in YAML_LIST:
            test_neck(os.path.join(path, yml_file))


if __name__ == '__main__':
    unittest.main()
