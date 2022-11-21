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

import cv2
import unittest
import yaml
import argparse

from ppcv.core.workspace import global_config
from ppcv.core.config import ConfigParser
from ppcv.engine.pipeline import Pipeline


class TestOcrDbDetOp(unittest.TestCase):
    def setUp(self):
        self.config = 'configs/unittest/test_ocr_db_det.yml'
        self.input = 'demo/00056221.jpg'
        self.cfg_dict = dict(config=self.config, input=self.input)
        cfg = argparse.Namespace(**self.cfg_dict)
        config = ConfigParser(cfg)
        config.print_cfg()
        self.model_cfg, self.env_cfg = config.parse()

    def test_detection(self):
        img = cv2.imread(self.input)
        inputs = [{"input.image": img, }]
        op_name = list(self.model_cfg[0].keys())[0]
        op = global_config[op_name](self.model_cfg[0][op_name], self.env_cfg)
        result = op(inputs)


class TestOcrCrnnRecOp(unittest.TestCase):
    def setUp(self):
        self.config = 'configs/unittest/test_ocr_crnn_rec.yml'
        self.input = 'demo/word_1.jpg'
        self.cfg_dict = dict(config=self.config, input=self.input)
        cfg = argparse.Namespace(**self.cfg_dict)
        config = ConfigParser(cfg)
        config.print_cfg()
        self.model_cfg, self.env_cfg = config.parse()

    def test_recognition(self):
        img = cv2.imread(self.input)
        inputs = [
            {
                "input.image": [img, img],
            },
            {
                "input.image": [img, img, img],
            },
            {
                "input.image": [img],
            },
        ]
        op_name = list(self.model_cfg[0].keys())[0]
        op = global_config[op_name](self.model_cfg[0][op_name], self.env_cfg)
        result = op(inputs)


class TestPPOCRv2(unittest.TestCase):
    def setUp(self):
        self.config = 'configs/system/PP-OCRv2.yml'
        self.input = 'demo/00056221.jpg'
        self.cfg_dict = dict(config=self.config, input=self.input)

    def test_ppocrv2(self):
        cfg = argparse.Namespace(**self.cfg_dict)
        input = os.path.abspath(self.input)
        pipeline = Pipeline(cfg)
        pipeline.run(input)


class TestPPOCRv3(unittest.TestCase):
    def setUp(self):
        self.config = 'configs/system/PP-OCRv3.yml'
        self.input = 'demo/00056221.jpg'
        self.cfg_dict = dict(config=self.config, input=self.input)

    def test_ppocrv3(self):
        cfg = argparse.Namespace(**self.cfg_dict)
        input = os.path.abspath(self.input)
        pipeline = Pipeline(cfg)
        pipeline.run(input)


class TestPPStructureTableStructureOp(unittest.TestCase):
    def setUp(self):
        self.config = 'configs/unittest/test_ocr_table_structure.yml'
        self.input = 'demo/table.jpg'
        self.cfg_dict = dict(config=self.config, input=self.input)
        cfg = argparse.Namespace(**self.cfg_dict)
        config = ConfigParser(cfg)
        config.print_cfg()
        self.model_cfg, self.env_cfg = config.parse()

    def test_table_structure(self):
        img = cv2.imread(self.input)
        inputs = [{"input.image": img, }]
        op_name = list(self.model_cfg[0].keys())[0]
        op = global_config[op_name](self.model_cfg[0][op_name], self.env_cfg)
        result = op(inputs)


class TestPPStructuretable(unittest.TestCase):
    def setUp(self):
        self.config = 'configs/system/PP-Structure-table.yml'
        self.input = 'demo/table.jpg'
        self.cfg_dict = dict(config=self.config, input=self.input)

    def test_structure_table(self):
        cfg = argparse.Namespace(**self.cfg_dict)
        input = os.path.abspath(self.input)
        pipeline = Pipeline(cfg)
        pipeline.run(input)


class TestLayoutDetectionOp(unittest.TestCase):
    def setUp(self):
        self.config = 'configs/unittest/test_ocr_layout.yml'
        self.input = 'demo/pp_structure_demo.png'
        self.cfg_dict = dict(config=self.config, input=self.input)
        cfg = argparse.Namespace(**self.cfg_dict)
        config = ConfigParser(cfg)
        config.print_cfg()
        self.model_cfg, self.env_cfg = config.parse()

    def test_layout_detection(self):
        img = cv2.imread(self.input)
        inputs = [{"input.image": img, }]
        op_name = list(self.model_cfg[0].keys())[0]
        op = global_config[op_name](self.model_cfg[0][op_name], self.env_cfg)
        result = op(inputs)


class TestPPStructure(unittest.TestCase):
    def setUp(self):
        self.config = 'configs/system/PP-Structure.yml'
        self.input = 'demo/pp_structure_demo.png'
        self.cfg_dict = dict(config=self.config, input=self.input)

    def test_ppstructure(self):
        cfg = argparse.Namespace(**self.cfg_dict)
        input = os.path.abspath(self.input)
        pipeline = Pipeline(cfg)
        pipeline.run(input)


class PPStructureKieSerOp(unittest.TestCase):
    def setUp(self):
        self.config = 'configs/system/PP-Structure-ser.yml'
        self.input = 'demo/kie_demo.jpg'
        self.cfg_dict = dict(config=self.config, input=self.input)
        cfg = argparse.Namespace(**self.cfg_dict)
        config = ConfigParser(cfg)
        config.print_cfg()
        self.model_cfg, self.env_cfg = config.parse()

    def test_ppstructure_kie_ser(self):
        img = cv2.imread(self.input)
        inputs = [{"input.image": img, }]
        op_name = list(self.model_cfg[0].keys())[0]
        op = global_config[op_name](self.model_cfg[0][op_name], self.env_cfg)
        result = op(inputs)


class PPStructureKieReOp(unittest.TestCase):
    def setUp(self):
        self.config = 'configs/system/PP-Structure-re.yml'
        self.input = 'demo/kie_demo.jpg'
        self.cfg_dict = dict(config=self.config, input=self.input)
        cfg = argparse.Namespace(**self.cfg_dict)
        config = ConfigParser(cfg)
        config.print_cfg()
        self.model_cfg, self.env_cfg = config.parse()

    def test_ppstructure_kie_re(self):
        img = cv2.imread(self.input)
        inputs = [{"input.image": img, }]
        op_name = list(self.model_cfg[0].keys())[0]
        op = global_config[op_name](self.model_cfg[0][op_name], self.env_cfg)
        result = op(inputs)


if __name__ == '__main__':
    unittest.main()
