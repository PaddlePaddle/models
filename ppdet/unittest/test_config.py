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


import os
import glob

import unittest
from ppdet.core.config import *


CFG_PATH = './configs'

class TestConfig(unittest.TestCase):

    def test_load_cfg(self):
        ymls = glob.glob(os.path.join(CFG_PATH, '*.yml'))
        for yml in ymls:
            cfg = load_cfg(yml)
            assert cfg is not None
            assert cfg.MODEL.TYPE is not None
            assert cfg.MODEL.BACKBONE is not None
            assert cfg.OPTIMIZER.TYPE is not None
            assert cfg.TRAIN.MAX_ITERS is not None


if __name__ == "__main__":
    unittest.main()
