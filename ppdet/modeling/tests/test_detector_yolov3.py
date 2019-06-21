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
import numpy as np

import paddle.fluid as fluid

from ppdet.modeling.tests.decorator_helper import prog_scope
from ppdet.core.config import load_cfg, merge_cfg
from ppdet.modeling.registry import Detectors


class TestDetectorYOLOv3(unittest.TestCase):
    def setUp(self):
        cfg_file = 'configs/yolov3_DarkNet53_1x_syncbn.yml'
        self.cfg = load_cfg(cfg_file)
        self.detector_type = 'YOLOv3'

    @prog_scope()
    def test_train(self):
        self.detector = Detectors.get(self.detector_type)(self.cfg)
        loss = self.detector.train()
        assert loss is not None

    @prog_scope()
    def test_test(self):
        self.detector = Detectors.get(self.detector_type)(self.cfg)
        pred = self.detector.test()
        assert pred is not None


if __name__ == '__main__':
    unittest.main()
