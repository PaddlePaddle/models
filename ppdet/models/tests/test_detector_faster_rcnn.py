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

from ppdet.models.tests.decorator_helper import prog_scope
from ppdet.core.workspace import load_config, merge_config, create
from tools.placeholder import create_feeds


class TestDetectorFasterRCNN(unittest.TestCase):
    def setUp(self):
        cfg_file = 'config_demo/faster_rcnn_r50_1x.yml'
        self.cfg = load_config(cfg_file)
        self.detector_type = self.cfg['architecture']

    @prog_scope()
    def test_train(self):
        train_feed = create(self.cfg['train_feed'])
        model = create(self.detector_type)
        _, feed_vars = create_feeds(train_feed)
        train_fetches = model.train(feed_vars)

    @prog_scope()
    def test_test(self):
        test_feed = create(self.cfg['test_feed'])
        model = create(self.detector_type)
        _, feed_vars = create_feeds(test_feed)
        test_fetches = model.test(feed_vars)


if __name__ == '__main__':
    unittest.main()
