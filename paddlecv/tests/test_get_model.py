#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import os
import sys
parent = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(parent, '../')))

import os
import paddle
import ppcv
import unittest

TASK_NAME = 'PP-YOLOE'
MODEL_NAME = 'paddlecv://models/PPLCNet_x1_0_infer/inference.pdiparams'


class TestGetConfigFile(unittest.TestCase):
    def test_main(self):
        try:
            cfg_file = ppcv.model_zoo.get_config_file(TASK_NAME)
            model_file = ppcv.model_zoo.get_model_file(MODEL_NAME)
            assert os.path.isfile(cfg_file)
            assert os.path.isfile(model_file)
        except:
            self.assertTrue(False)


if __name__ == '__main__':
    unittest.main()
