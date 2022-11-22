# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved. 
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
import os
import sys
parent = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(parent, '../')))

import unittest
from ppcv.engine.pipeline import Pipeline
import yaml
import argparse


class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.config = 'configs/unittest/test_pipeline.yml'
        self.input = 'demo/ILSVRC2012_val_00020010.jpeg'
        self.cfg_dict = dict(config=self.config, input=self.input)

    def test_pipeline(self):
        cfg = argparse.Namespace(**self.cfg_dict)
        input = os.path.abspath(self.input)
        pipeline = Pipeline(cfg)
        pipeline.run(input)


if __name__ == '__main__':
    unittest.main()
