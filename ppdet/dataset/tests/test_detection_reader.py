import os
import time
import unittest
import sys
import logging
import numpy as np
import yaml

import set_env
from dataset import DetectionReader as DetReader

logging.basicConfig(level=logging.INFO)
class TestDetectionReader(unittest.TestCase):
    """Test cases for dataset.detection_reader
    """
    @classmethod
    def setUpClass(cls):
        """ setup
        """
        prefix = os.path.dirname(os.path.abspath(__file__))
        coco_yml = os.path.join(prefix, 'coco.yml')
        with open(coco_yml, 'rb') as f:
            cls.coco_conf = yaml.load(f.read())

    @classmethod
    def tearDownClass(cls):
        """ tearDownClass """
        pass

    def test_train(self):
        """ Test reader for training
        """
        coco = DetReader(self.coco_conf['DATA'],
            self.coco_conf['TRANSFORM'])
        train_rd = coco.train()
        self.assertTrue(train_rd is not None)

        ct = 0
        for sample in train_rd():
            ct += 1
            self.assertTrue(sample is not None)
        self.assertEqual(ct, self.coco_conf['DATA']['samples'])

    def test_val(self):
        """ Test reader for validation
        """
        coco = DetReader(self.coco_conf['DATA'],
            self.coco_conf['TRANSFORM'])
        val_rd = coco.val()
        self.assertTrue(val_rd is not None)

        # test 3 epoches
        for _ in range(3):
            ct = 0
            for sample in val_rd():
                ct += 1
                self.assertTrue(sample is not None)
            self.assertEqual(ct, self.coco_conf['DATA']['samples'])


if __name__ == '__main__':
    unittest.main()
