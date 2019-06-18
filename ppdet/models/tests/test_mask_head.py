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

import unittest

import paddle.fluid as fluid

from ppdet.core.config import load_cfg, merge_cfg
from ppdet.models.tests.decorator_helper import prog_scope
from ppdet.models.bbox_heads.mask_head import MaskHead

YAML_LIST = [
    'configs/mask-rcnn_ResNet50-C4_1x.yml',
    'configs/mask-rcnn_ResNet50-C4_2x.yml',
]


def get_datas():
    roi_feat = fluid.layers.data(
        name='roi_feat', shape=[1024, 14, 14], dtype='float32', lod_level=1)
    mask_int32 = fluid.layers.data(
        name='mask_int32', shape=[1], dtype='int32', lod_level=1)
    bbox_pred = fluid.layers.data(
        name='bbox_pred', shape=[6], dtype='float32', lod_level=1)
    return roi_feat, mask_int32, bbox_pred


class TestMaskHead(unittest.TestCase):
    @prog_scope()
    def build_loss(self, cfg_file):
        cfg = load_cfg(cfg_file)
        merge_cfg({'IS_TRAIN': True}, cfg)
        mask_head = MaskHead(cfg)
        roi_feat, mask_int32, bbox_pred = get_datas()
        loss = mask_head.get_loss(roi_feat, mask_int32)
        loss = loss['loss_mask']
        assert loss is not None
        assert loss.shape == (1, )

    def test_loss(self):
        for cfg_file in YAML_LIST:
            self.build_loss(cfg_file)

    @prog_scope()
    def build_prediction(self, cfg_file):
        cfg = load_cfg(cfg_file)
        merge_cfg({'IS_TRAIN': False}, cfg)
        mask_head = MaskHead(cfg)
        roi_feat, mask_int32, bbox_pred = get_datas()
        pred = mask_head.get_prediction(roi_feat, bbox_pred)
        assert pred is not None
        assert pred.shape == (-1, 81, 28, 28)

    def test_prediction(self):
        for cfg_file in YAML_LIST:
            self.build_prediction(cfg_file)


if __name__ == '__main__':
    unittest.main()
