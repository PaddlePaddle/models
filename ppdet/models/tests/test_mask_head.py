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
import ppdet.models.mask_head as mask_head
import os

YAML_LIST = [
    'mask-rcnn_ResNet50-C4_1x.yml',
    'mask-rcnn_ResNet50-C4_2x.yml',
]


def init_head_input(cfg, mode):
    head_inputs = []
    roi_feat = fluid.layers.data(
        name='roi_feat', shape=[1024, 14, 14], dtype='float32', lod_level=1)
    rois = fluid.layers.data(
        name='rois', shape=[4], dtype='float32', lod_level=1)
    labels_int32 = fluid.layers.data(
        name='labels_int32', shape=[1], dtype='int32', lod_level=1)
    bbox_det = fluid.layers.data(
        name='bbox_det', shape=[6], dtype='float32', lod_level=1)
    gt_label = fluid.layers.data(
        name='gt_label', shape=[1], dtype='int32', lod_level=1)
    is_crowd = fluid.layers.data(
        name='is_crowd', shape=[1], dtype='int32', lod_level=1)
    gt_box = fluid.layers.data(
        name='gt_box', shape=[4], dtype='float32', lod_level=1)
    im_info = fluid.layers.data(name='im_info', shape=[3], dtype='float32')
    gt_segms = fluid.layers.data(
        name='gt_segms', shape=[2], dtype='float32', lod_level=3)
    if mode == "train":
        mask_head_func = cfg.MASK_HEAD.TRAIN.HEAD_FUNC
    else:
        mask_head_func = cfg.MASK_HEAD.TEST.HEAD_FUNC
    head_func = getattr(mask_head, mask_head_func)
    head_inputs.append(roi_feat)
    head_inputs.append(rois)
    head_inputs.append(labels_int32)
    head_inputs.append(bbox_det)
    head_inputs.append(gt_label)
    head_inputs.append(is_crowd)
    head_inputs.append(gt_box)
    head_inputs.append(im_info)
    head_inputs.append(gt_segms)
    head_inputs.append(head_func)
    return head_inputs


def test_mask_train_head(cfg_file):
    cfg = load_cfg(cfg_file)
    main_program = Program()
    start_program = Program()
    with program_guard(main_program, start_program):
        ob = mask_head.MaskHead(cfg)
        head_inputs = init_head_input(cfg, 'train')
        roi_feat = head_inputs[0]
        rois = head_inputs[1]
        labels_int32 = head_inputs[2]
        gt_label = head_inputs[4]
        is_crowd = head_inputs[5]
        gt_box = head_inputs[6]
        im_info = head_inputs[7]
        gt_segms = head_inputs[8]
        head_func = head_inputs[9]
        mask_rois, roi_has_mask_int32, mask_int32 = ob.get_target(
            rois, labels_int32, gt_label, is_crowd, gt_box, im_info, gt_segms)
        mask_fcn_logits = ob.get_output(roi_feat, head_func, 'train')

        loss_mask = ob.get_loss(mask_int32, mask_fcn_logits)

        assert mask_fcn_logits is not None
        assert mask_rois is not None
        assert roi_has_mask_int32 is not None
        assert mask_int32 is not None
        assert loss_mask is not None


def test_mask_test_head(cfg_file):
    cfg = load_cfg(cfg_file)
    main_program = Program()
    start_program = Program()
    with program_guard(main_program, start_program):
        ob = mask_head.MaskHead(cfg)
        head_inputs = init_head_input(cfg, 'test')
        roi_feat = head_inputs[0]
        bbox_det = head_inputs[3]
        head_func = head_inputs[9]
        mask_det = ob.get_prediction(bbox_det, roi_feat, head_func)

        assert mask_det is not None


class TestMaskHead(unittest.TestCase):
    def test_mask_train_heads(self):
        path = os.path.dirname(configs.__file__)
        for yml_file in YAML_LIST:
            test_mask_train_head(os.path.join(path, yml_file))

    def test_mask_test_heads(self):
        path = os.path.dirname(configs.__file__)
        for yml_file in YAML_LIST:
            test_mask_test_head(os.path.join(path, yml_file))


if __name__ == '__main__':
    unittest.main()
