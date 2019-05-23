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
import ppdet.models.bbox_heads as bbox_head
import ppdet.models.target_assigners as bbox_assigner
import os

YAML_LIST = [
    'faster-rcnn_ResNet50-C4_1x.yml',
    'faster-rcnn_ResNet50-C4_2x.yml',
    'mask-rcnn_ResNet50-C4_1x.yml',
    'mask-rcnn_ResNet50-C4_2x.yml',
]


def build_feed_vars():
    feed_info = [
        {
            'name': 'roi_feat',
            'shape': [1024, 14, 14],
            'dtype': 'float32',
            'lod_level': 1
        },
        {
            'name': 'rpn_rois',
            'shape': [4],
            'dtype': 'float32',
            'lod_level': 1
        },
        {
            'name': 'im_info',
            'shape': [3],
            'dtype': 'float32',
            'lod_level': 0
        },
        {
            'name': 'gt_box',
            'shape': [1],
            'dtype': 'float32',
            'lod_level': 1
        },
        {
            'name': 'gt_label',
            'shape': [1],
            'dtype': 'int32',
            'lod_level': 1
        },
        {
            'name': 'is_crowd',
            'shape': [1],
            'dtype': 'int32',
            'lod_level': 1
        },
    ]
    feed_vars = {}
    for info in feed_info:
        d = fluid.layers.data(
            name=info['name'],
            shape=info['shape'],
            dtype=info['dtype'],
            lod_level=info['lod_level'])
        feed_vars[info['name']] = d
    return feed_vars


def test_bbox_head(cfg_file):
    cfg = load_cfg(cfg_file)
    main_program = Program()
    start_program = Program()
    with program_guard(main_program, start_program):
        ob = bbox_head.BBoxHead(cfg)
        assigner = bbox_assigner.BBoxAssigner(cfg)
        feed_vars = build_feed_vars()
        rpn_rois = feed_vars['rpn_rois']
        roi_feat = feed_vars['roi_feat']
        im_info = feed_vars['im_info']
        assign_output = assigner.get_sampled_rois_and_targets(rpn_rois,
                                                              feed_vars)
        labels_int32 = assign_output[1]
        bbox_targets = assign_output[2]
        bbox_inside_weights = assign_output[3]
        bbox_outside_weights = assign_output[4]
        loss_dict = ob.get_loss(roi_feat, labels_int32, bbox_targets,
                                bbox_inside_weights, bbox_outside_weights)
        loss_cls = loss_dict['loss_cls']
        loss_bbox = loss_dict['loss_cls']
        pred_result = ob.get_prediction(roi_feat, rpn_rois, im_info)

        assert labels_int32 is not None
        assert bbox_targets is not None
        assert bbox_inside_weights is not None
        assert bbox_outside_weights is not None
        assert loss_cls is not None
        assert loss_bbox is not None
        assert pred_result is not None


class TestBBoxHead(unittest.TestCase):
    def test_bbox_heads(self):
        path = os.path.dirname(configs.__file__)
        for yml_file in YAML_LIST:
            test_bbox_head(os.path.join(path, yml_file))


if __name__ == '__main__':
    unittest.main()
