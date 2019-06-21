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
# TODO(luoqianhui): change comment stype above in github

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

from ppdet.modeling.tests.decorator_helper import prog_scope
from ppdet.core.config import load_cfg, merge_cfg
from ppdet.modeling.registry import Detectors


class TestDetectorRetinaNet(unittest.TestCase):
    """
    Test the detector: retinanet
    """

    def setUp(self):
        cfg_file = 'configs/retinanet_ResNet50-FPN_1x.yml'
        self.cfg = load_cfg(cfg_file)
        self.detector_type = 'RetinaNet'

    @prog_scope()
    def test_train(self):
        """
        Test the training stage of retinanet
        """
        merge_cfg({'IS_TRAIN': True}, self.cfg)
        self.detector = Detectors.get(self.detector_type)(self.cfg)
        self.detector.train()
        # TODO(luoqianhui): add more check

    @prog_scope()
    def test_test(self):
        """
        Test the testing stage of retinanet
        """
        merge_cfg({'IS_TRAIN': False}, self.cfg)
        self.detector = Detectors.get(self.detector_type)(self.cfg)
        self.detector.test()
        # TODO(luoqianhui): add more check


if __name__ == '__main__':
    unittest.main()
