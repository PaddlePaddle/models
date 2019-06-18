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
import ppdet.models.roi_extractors as roi_extractors
from ppdet.core.config import load_cfg
import configs
import paddle.fluid as fluid
from ppdet.models.tests.decorator_helper import prog_scope
import os

YAML_LIST = [
    'faster-rcnn_ResNet50-C4_1x.yml',
    'faster-rcnn_ResNet50-C4_2x.yml',
    'mask-rcnn_ResNet50-C4_1x.yml',
    'mask-rcnn_ResNet50-C4_2x.yml',
    'faster-rcnn_ResNet50-FPN_1x.yml',
    'faster-rcnn_ResNet50-FPN_2x.yml',
]


def init_head_input(cfg, FPN_ON):
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
    if not FPN_ON:
        return head_inputs[-1]
    name_list = ['res' + str(lvl) + '_sum' for lvl in range(2, 6)]
    head_dict = {k: v for k, v in zip(name_list, head_inputs)}
    return head_dict, name_list


@prog_scope()
def test_roi_extractor(cfg_file):
    cfg = load_cfg(cfg_file)
    method = getattr(roi_extractors, cfg.ROI_EXTRACTOR.EXTRACT_METHOD)
    FPN_ON = getattr(cfg.MODEL, 'NECK', False)
    ob = method(cfg)
    rois = fluid.layers.data(name='rois', shape=[4], dtype='int32', lod_level=1)
    if FPN_ON:
        head_dict, name_list = init_head_input(cfg, True)
        spatial_scale = [1. / 32., 1. / 16., 1. / 8., 1. / 4.]
        roi_feat = ob.get_roi_feat(head_dict, rois, name_list, spatial_scale)
    else:
        head_input = init_head_input(cfg, False)
        roi_feat = ob.get_roi_feat(head_input, rois)

    assert roi_feat is not None


class TestRoIExtractor(unittest.TestCase):
    def test_roi_extractors(self):
        path = os.path.dirname(configs.__file__)
        for yml_file in YAML_LIST:
            test_roi_extractor(os.path.join(path, yml_file))


if __name__ == "__main__":
    unittest.main()
