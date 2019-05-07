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
import ppdet.models.bbox_head as bbox_head
import os

YAML_LIST = [
    'faster-rcnn_ResNet50-C4_1x.yml',
    'faster-rcnn_ResNet50-C4_2x.yml',
    'mask-rcnn_ResNet50-C4_1x.yml',
    'mask-rcnn_ResNet50-C4_2x.yml',
]


def init_input(cfg):
    out = [cfg]
    gt_label = fluid.layers.data(
        name='gt_label', shape=[1], dtype='int32', lod_level=1)
    is_crowd = fluid.layers.data(
        name='is_crowd', shape=[1], dtype='int32', lod_level=1)
    gt_box = fluid.layers.data(
        name='gt_box', shape=[4], dtype='float32', lod_level=1)
    im_info = fluid.layers.data(name='im_info', shape=[3], dtype='float32')
    head_func = getattr(bbox_head, cfg.BBOX_HEAD.HEAD_FUNC)
    out.append(gt_label)
    out.append(is_crowd)
    out.append(gt_box)
    out.append(im_info)
    out.append(head_func)
    return out


def init_head_input():
    roi_feat = fluid.layers.data(
        name='roi_feat', shape=[1024, 14, 14], dtype='float32', lod_level=1)
    rpn_rois = fluid.layers.data(
        name='rpn_rois', shape=[4], dtype='float32', lod_level=1)
    return roi_feat, rpn_rois


def test_bbox_head(cfg_file):
    cfg = load_cfg(cfg_file)
    main_program = Program()
    start_program = Program()
    with program_guard(main_program, start_program):
        out = init_input(cfg)
        ob = bbox_head.BBoxHead(*out)
        roi_feat, rpn_rois = init_head_input()
        cls_score, bbox_pred = ob.get_output(roi_feat)
        rois, labels_int32, bbox_targets, bbox_inside_weights, bbox_outside_weights = ob.get_target(
            rpn_rois)
        loss_cls, loss_bbox = ob.get_loss(cls_score, bbox_pred, labels_int32,
                                          bbox_targets, bbox_inside_weights,
                                          bbox_outside_weights)
        pred_result = ob.get_prediction(rpn_rois, cls_score, bbox_pred)

        assert cls_score is not None
        assert bbox_pred is not None
        assert rois is not None
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
