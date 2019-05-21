import os
import time
import unittest
import sys
import logging
import numpy as np
import yaml

import set_env
from dataset import Reader

logging.basicConfig(level=logging.INFO)
class TestReader(unittest.TestCase):
    """Test cases for dataset.reader
    """
    @classmethod
    def setUpClass(cls):
        """ setup
        """
        prefix = os.path.dirname(os.path.abspath(__file__))
        coco_yml = os.path.join(prefix, 'coco.yml')
        with open(coco_yml, 'rb') as f:
            cls.coco_conf = yaml.load(f.read())
        rcnn_yml = os.path.join(prefix, 'rcnn_dataset.yml')
        with open(rcnn_yml, 'rb') as f:
            cls.rcnn_conf = yaml.load(f.read())
        devices = os.getenv("CUDA_VISIBLE_DEVICES") or ""
        cls.devices_num = len(devices.split(","))

    @classmethod
    def tearDownClass(cls):
        """ tearDownClass """
        pass

    def test_train(self):
        """ Test reader for training
        """
        coco = Reader(self.coco_conf['DATA'],
            self.coco_conf['TRANSFORM'])
        train_rd = coco.train()
        self.assertTrue(train_rd is not None)

        ct = 0
        for sample in train_rd():
            ct += 1
            self.assertTrue(sample is not None)
        batch_size = self.coco_conf['TRANSFORM']['batch_size'] * self.devices_num
        if self.coco_conf['DATA']['samples']%batch_size == 0 \
                or 'drop_last' not in self.coco_conf['TRANSFORM'] \
                or self.coco_conf['TRANSFORM']['drop_last'] == True:
            self.assertEqual(ct, int(self.coco_conf['DATA']['samples']/batch_size)*self.devices_num)
        elif 'drop_last' in self.coco_conf['TRANSFORM'] and self.coco_conf['TRANSFORM']['drop_last'] == False:
            self.assertGreaterEqual(ct, int(self.coco_conf['DATA']['samples']/batch_size)*self.devices_num) 
            

    def test_val(self):
        """ Test reader for validation
        """
        coco = Reader(self.coco_conf['DATA'],
            self.coco_conf['TRANSFORM'])
        val_rd = coco.val()
        self.assertTrue(val_rd is not None)

        # test 3 epoches
        for _ in range(3):
            ct = 0
            for sample in val_rd():
                ct += 1
                self.assertTrue(sample is not None)
            batch_size = self.coco_conf['TRANSFORM']['batch_size'] * self.devices_num
            if self.coco_conf['DATA']['samples']%batch_size == 0 \
                    or 'drop_last' not in self.coco_conf['TRANSFORM'] \
                    or self.coco_conf['TRANSFORM']['drop_last'] == True:
                self.assertEqual(ct, int(self.coco_conf['DATA']['samples']/batch_size)*self.devices_num) 
            elif 'drop_last' in self.coco_conf['TRANSFORM'] and self.coco_conf['TRANSFORM']['drop_last'] == False:
                self.assertGreateEqual(ct, int(self.coco_conf['DATA']['samples']/batch_size)*self.devices_num)
     
    def test_rcnn(self):
        """ Test reader for training
        """
        rcnn = Reader(self.rcnn_conf['DATA'],
            self.rcnn_conf['TRANSFORM'])
        rcnn_rd = rcnn.train()
        self.assertTrue(rcnn_rd is not None)

        ct = 0
        out = None
        for sample in rcnn_rd():
            out = sample
            ct += 1
            self.assertTrue(sample is not None)
        self.assertEqual(out[0][1].shape[1], 4)
        self.assertEqual(out[0][2].shape[1], 1)
        self.assertEqual(out[0][3].shape[1], 1)
        self.assertEqual(out[0][4].shape[0], 3)
        self.assertEqual(out[0][5].shape[0], 1)
        batch_size = self.rcnn_conf['TRANSFORM']['batch_size'] * self.devices_num
        if self.rcnn_conf['DATA']['samples']%batch_size == 0 \
                or 'drop_last' not in self.rcnn_conf['TRANSFORM'] \
                or self.rcnn_conf['TRANSFORM']['drop_last'] == True:
            self.assertEqual(ct, int(self.rcnn_conf['DATA']['samples']/batch_size)*self.devices_num)
        elif 'drop_last' in self.rcnn_conf['TRANSFORM'] and self.rcnn_conf['TRANSFORM']['drop_last'] == False:
            self.assertGreateEqual(ct, int(self.rcnn_conf['DATA']['samples']/batch_size)*self.devices_num) 


if __name__ == '__main__':
    unittest.main()
