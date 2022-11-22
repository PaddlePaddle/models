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

import numpy as np
import cv2
import copy
import unittest
import yaml
import argparse

from ppcv.core.workspace import global_config
from ppcv.core.config import ConfigParser


class TestClsCorrection(unittest.TestCase):
    def setUp(self):
        self.config = 'configs/unittest/test_cls_connector.yml'
        self.input = 'demo/ILSVRC2012_val_00020010.jpeg'
        self.cfg_dict = dict(config=self.config, input=self.input)
        cfg = argparse.Namespace(**self.cfg_dict)
        config = ConfigParser(cfg)
        config.print_cfg()
        self.model_cfg, self.env_cfg = config.parse()

    def test_cls_correction(self):
        img = cv2.imread(self.input)[:, :, ::-1]
        inputs = [
            {
                "input.image": [img],
                "input.class_ids": [[0], ],
                "input.scores": [[0.95]],
            },
            {
                "input.image": [img, img],
                "input.class_ids": [[1], [3]],
                "input.scores": [[0.95], [0.85]],
            },
        ]
        op_name = "ClsCorrectionOp"
        op = global_config[op_name](self.model_cfg[0][op_name])
        result = op(inputs)
        self.assert_equal(img, result)

    def assert_equal(self, img, result):
        diff = np.sum(np.abs(img - result[0][0]))
        self.assertEqual(diff, 0)

        corr_img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        diff = np.sum(np.abs(corr_img - result[1][0]))
        self.assertEqual(diff, 0)

        diff = np.sum(np.abs(img - result[1][1]))
        self.assertEqual(diff, 0)


class TestBboxCropOp(unittest.TestCase):
    def setUp(self):
        self.config = 'configs/unittest/test_bbox_crop.yml'
        self.input = 'demo/ILSVRC2012_val_00020010.jpeg'
        self.cfg_dict = dict(config=self.config, input=self.input)
        cfg = argparse.Namespace(**self.cfg_dict)
        config = ConfigParser(cfg)
        config.print_cfg()
        self.model_cfg, self.env_cfg = config.parse()

    def test_bbox_crop(self):
        img = cv2.imread(self.input)[:, :, ::-1]
        inputs = [
            {
                "input.image": img,
                "input.bbox": np.array([[1, 1, 2, 5]]),
            },
            {
                "input.image": img,
                "input.bbox": np.array([[1, 1, 10, 20]]),
            },
        ]

        op_name = list(self.model_cfg[0].keys())[0]
        op = global_config[op_name](self.model_cfg[0][op_name])
        result = op(inputs)
        self.assert_equal(result)

    def assert_equal(self, result):
        gt_res = 55398.0
        sums = 0
        for idx, r in enumerate(result):
            for k, poly in r.items():
                sums += np.sum(np.abs(poly))
        self.assertEqual(gt_res, sums)


class TestPolyCropOp(unittest.TestCase):
    def setUp(self):
        self.config = 'configs/unittest/test_poly_crop.yml'
        self.input = 'demo/ILSVRC2012_val_00020010.jpeg'
        self.cfg_dict = dict(config=self.config, input=self.input)
        cfg = argparse.Namespace(**self.cfg_dict)
        config = ConfigParser(cfg)
        config.print_cfg()
        self.model_cfg, self.env_cfg = config.parse()

    def test_poly_crop(self):
        img = cv2.imread(self.input)[:, :, ::-1]
        inputs = [
            {
                "input.image": img,
                "input.poly": np.array(
                    [[[1, 1], [2, 1], [2, 3], [1, 3]]]).astype(np.float32),
            },
            {
                "input.image": img,
                "input.poly": np.array(
                    [[[1, 1], [2, 1], [2, 3], [0, 3]]]).astype(np.float32),
            },
        ]
        op_name = list(self.model_cfg[0].keys())[0]
        op = global_config[op_name](self.model_cfg[0][op_name])
        result = op(inputs)
        self.assert_equal(result)

    def assert_equal(self, result):
        gt_res = 3620.0
        sums = 0
        for idx, r in enumerate(result):
            for poly in r['poly_crop.crop_image']:
                sums += np.sum(np.abs(poly))
        self.assertEqual(gt_res, sums)


class TestFragmentCompositionOp(unittest.TestCase):
    def setUp(self):
        self.config = 'configs/unittest/test_fragment_composition.yml'
        self.input = 'demo/ILSVRC2012_val_00020010.jpeg'
        self.cfg_dict = dict(config=self.config, input=self.input)
        cfg = argparse.Namespace(**self.cfg_dict)
        config = ConfigParser(cfg)
        config.print_cfg()
        self.model_cfg, self.env_cfg = config.parse()

    def test_fragment_composition(self):
        inputs = [
            {
                "input.text": ["hello", "world"]
            },
            {
                "input.text": ["paddle", "paddle"]
            },
        ]
        op_name = list(self.model_cfg[0].keys())[0]
        op = global_config[op_name](self.model_cfg[0][op_name])
        result = op(inputs)

        self.assert_equal(result)

    def assert_equal(self, result):
        gt_res = ["hello world", "paddle paddle"]
        for idx, r in enumerate(result):
            self.assertEqual(gt_res[idx], r)


class TestKeyFrameExtractionOp(unittest.TestCase):
    def setUp(self):
        self.config = 'configs/unittest/test_key_frame_extraction.yml'
        self.input = 'demo/pikachu.mp4'
        self.cfg_dict = dict(config=self.config, input=self.input)
        cfg = argparse.Namespace(**self.cfg_dict)
        config = ConfigParser(cfg)
        config.print_cfg()
        self.model_cfg, self.env_cfg = config.parse()

    def test_key_frame_extraction(self):
        inputs = [{"input.video_path": "demo/pikachu.mp4", }]
        op_name = list(self.model_cfg[0].keys())[0]
        op = global_config[op_name](self.model_cfg[0][op_name])
        result = op(inputs)

        self.assert_equal(result)

    def assert_equal(self, result):
        key_frames_id = [
            7, 80, 102, 139, 200, 234, 271, 320, 378, 437, 509, 592, 619, 684,
            749, 791, 843, 872, 934, 976, 1028, 1063, 1156, 1179, 1249, 1356,
            1400, 1461, 1516, 1630, 1668, 1718, 1768
        ]
        gt_abs_sum = 2312581162.0
        abs_sum = sum([np.sum(np.abs(k)) for k in result[0][0]])
        self.assertEqual(result[0][1], key_frames_id)
        self.assertAlmostEqual(gt_abs_sum, abs_sum)


class TestTableMatcherOp(unittest.TestCase):
    def setUp(self):
        self.config = 'configs/unittest/test_table_matcher.yml'
        self.input = './demo/table_demo.npy'
        self.cfg_dict = dict(config=self.config, input=self.input)
        cfg = argparse.Namespace(**self.cfg_dict)
        config = ConfigParser(cfg)
        config.print_cfg()
        self.model_cfg, self.env_cfg = config.parse()

    def test_table_matcher(self):
        inputs = np.load(self.input, allow_pickle=True).item()

        gt_res = inputs['outputs']
        inputs = inputs['inputs']
        pipe_inputs = [{}]
        for k, v in inputs[0].items():
            pipe_inputs[0].update({'input.' + k: v})

        op_name = list(self.model_cfg[0].keys())[0]
        op = global_config[op_name](self.model_cfg[0][op_name])
        result = op(pipe_inputs)

        self.assertEqual(gt_res[0]['Matcher.html'],
                         result[0]['tablematcher.html'])


class TestPPStructureFilterOp(unittest.TestCase):
    def setUp(self):
        self.config = 'configs/unittest/test_ppstructure_filter.yml'
        self.input = 'demo/ILSVRC2012_val_00020010.jpeg'
        self.cfg_dict = dict(config=self.config, input=self.input)
        cfg = argparse.Namespace(**self.cfg_dict)
        config = ConfigParser(cfg)
        config.print_cfg()
        self.model_cfg, self.env_cfg = config.parse()

    def test_ppstructure_filter(self):
        inputs = [
            {
                "input.dt_cls_names": ['table', 'txt'],
                "input.crop_image": ['', '1'],
                "input.dt_polys": [1, 0],
                "input.rec_text": ['a', 'b']
            },
            {
                "input.dt_cls_names": ['table', 'txt'],
                "input.crop_image": ['1', ''],
                "input.dt_polys": [0, 1],
                "input.rec_text": ['b', 'a']
            },
        ]
        gt_res = [
            {
                "filter.image": [''],
                "filter.dt_polys": [1],
                "filter.rec_text": ['a']
            },
            {
                "filter.image": ['1'],
                "filter.dt_polys": [0],
                "filter.rec_text": ['b']
            },
        ]

        op_name = list(self.model_cfg[0].keys())[0]
        op = global_config[op_name](self.model_cfg[0][op_name])
        result = op(inputs)

        self.assertEqual(gt_res, result)


class TestPPStructureResultConcatOp(unittest.TestCase):
    def setUp(self):
        self.config = 'configs/unittest/test_ppstructure_result_concat.yml'
        self.input = 'demo/ILSVRC2012_val_00020010.jpeg'
        self.cfg_dict = dict(config=self.config, input=self.input)
        cfg = argparse.Namespace(**self.cfg_dict)
        config = ConfigParser(cfg)
        config.print_cfg()
        self.model_cfg, self.env_cfg = config.parse()

    def test_ppstructure_result_concat(self):
        inputs = [
            {
                "input.table.structures": [['td'], ['td']],
                "input.Matcher.html": ['html', 'html'],
                "input.layout.dt_bboxes": [0, 1, 2, 3, 4],
                "input.table.dt_bboxes": [[0], [1]],
                "input.filter_table.dt_polys": [[0], [1]],
                "input.filter_table.rec_text": [['0'], ['1']],
                "input.filter_txts.dt_polys": [[2], [3], [4]],
                "input.filter_txts.rec_text": [['2'], ['3'], ['4']],
            },
            {
                "input.table.structures": [['td'], ['td']],
                "input.Matcher.html": ['html', 'html'],
                "input.layout.dt_bboxes": [5, 1, 2, 3, 4],
                "input.table.dt_bboxes": [[5], [1]],
                "input.filter_table.dt_polys": [[5], [1]],
                "input.filter_table.rec_text": [['5'], ['1']],
                "input.filter_txts.dt_polys": [[2], [3], [4]],
                "input.filter_txts.rec_text": [['2'], ['3'], ['4']],
            },
        ]
        gt_res = [{
            'concat.dt_polys': [[2], [3], [4], [0], [1]],
            'concat.rec_text': [['2'], ['3'], ['4'], ['0'], ['1']],
            'concat.dt_bboxes': [0, 1, 2, 3, 4],
            'concat.html': ['', '', '', 'html', 'html'],
            'concat.cell_bbox': [[], [], [], [0], [1]],
            'concat.structures': [[], [], [], ['td'], ['td']]
        }, {
            'concat.dt_polys': [[2], [3], [4], [5], [1]],
            'concat.rec_text': [['2'], ['3'], ['4'], ['5'], ['1']],
            'concat.dt_bboxes': [5, 1, 2, 3, 4],
            'concat.html': ['', '', '', 'html', 'html'],
            'concat.cell_bbox': [[], [], [], [5], [1]],
            'concat.structures': [[], [], [], ['td'], ['td']]
        }]

        op_name = list(self.model_cfg[0].keys())[0]
        op = global_config[op_name](self.model_cfg[0][op_name])
        result = op(inputs)

        self.assertEqual(gt_res, result)


if __name__ == '__main__':
    unittest.main()
