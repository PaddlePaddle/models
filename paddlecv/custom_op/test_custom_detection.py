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
import cv2
import unittest
import yaml
import argparse

import paddlecv
from ppcv.core.workspace import global_config
from ppcv.core.config import ConfigParser


class TestCustomDetection(unittest.TestCase):
    def setUp(self):
        self.config = 'test_custom_detection.yml'
        self.input = '../demo/ILSVRC2012_val_00020010.jpeg'
        self.cfg_dict = dict(config=self.config, input=self.input)
        cfg = argparse.Namespace(**self.cfg_dict)
        config = ConfigParser(cfg)
        config.print_cfg()
        self.model_cfg, self.env_cfg = config.parse()

    def test_detection(self):
        img = cv2.imread(self.input)[:, :, ::-1]
        inputs = [
            {
                "input.image": img,
            },
            {
                "input.image": img,
            },
        ]
        op_name = list(self.model_cfg[0].keys())[0]
        det_op = global_config[op_name](self.model_cfg[0][op_name],
                                        self.env_cfg)
        result = det_op(inputs)

    def test_pipeline(self):
        input = os.path.abspath(self.input)
        ppcv = paddlecv.PaddleCV(config_path=self.config)
        ppcv(input)


if __name__ == '__main__':
    unittest.main()
