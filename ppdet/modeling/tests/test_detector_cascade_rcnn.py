"""
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
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import numpy as np

import paddle.fluid as fluid

from ppdet.modeling.tests.decorator_helper import prog_scope
from ppdet.core.config import load_cfg, merge_cfg
from ppdet.modeling.registry import Detectors


class TestDetectorCascadeRCNN(unittest.TestCase):
    """
    CacadeRCNN's unit-test
    """

    def setUp(self):
        cfg_file = 'configs/debug_cascade-rcnn_ResNet50-FPN_1x.yml'
        self.cfg = load_cfg(cfg_file)
        self.detector_type = 'CascadeRCNN'

    @prog_scope()
    def test_train(self):
        """
	test train process
	"""
        merge_cfg({'IS_TRAIN': True}, self.cfg)
        assert self.cfg.IS_TRAIN
        self.detector = Detectors.get(self.detector_type)(self.cfg)
        self.detector.train()

    #print(fluid.default_main_program())
    #TODO(dangqingqing): add more check

    @prog_scope()
    def test_test(self):
        """
	test infer process
	"""
        merge_cfg({'IS_TRAIN': False}, self.cfg)
        assert not self.cfg.IS_TRAIN
        self.detector = Detectors.get(self.detector_type)(self.cfg)
        self.detector.test()

    #TODO(dangqingqing): add more check


if __name__ == '__main__':
    unittest.main()
