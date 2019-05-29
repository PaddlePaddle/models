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

    @classmethod
    def tearDownClass(cls):
        """ tearDownClass """
        pass

    def test_train(self):
        """ Test reader for training
        """
        coco = Reader(self.coco_conf['DATA'], self.coco_conf['TRANSFORM'], 10)
        self.devices_num = self.coco_conf['TRANSFORM']['TRAIN']['DEVICES_NUM']
        train_rd = coco.train()
        self.assertTrue(train_rd is not None)

        ct = 0
        for sample in train_rd():
            ct += 1
            self.assertTrue(sample is not None)
        self.assertGreaterEqual(ct, coco._maxiter)

    def test_val(self):
        """ Test reader for validation
        """
        coco = Reader(self.coco_conf['DATA'], self.coco_conf['TRANSFORM'], 10)
        self.devices_num = self.coco_conf['TRANSFORM']['TRAIN']['DEVICES_NUM']
        val_rd = coco.val()
        self.assertTrue(val_rd is not None)

        # test 3 epoches
        for _ in range(3):
            ct = 0
            for sample in val_rd():
                ct += 1
                self.assertTrue(sample is not None)
            self.assertGreaterEqual(ct, coco._maxiter)

    def test_rcnn(self):
        """ Test reader for training
        """
        rcnn = Reader(self.rcnn_conf['DATA'], self.rcnn_conf['TRANSFORM'], 10)
        rcnn_rd = rcnn.train()
        self.assertTrue(rcnn_rd is not None)

        ct = 0
        out = None
        for sample in rcnn_rd():
            out = sample
            ct += 1
            self.assertTrue(sample is not None)
        self.assertEqual(out[0][1].shape[0], 3)
        self.assertEqual(out[0][2].shape[1], 4)
        self.assertEqual(out[0][3].shape[1], 1)
        self.assertEqual(out[0][4].shape[1], 1)
        self.assertEqual(out[0][5].shape[0], 1)
        self.assertGreaterEqual(ct, rcnn._maxiter)


if __name__ == '__main__':
    unittest.main()
