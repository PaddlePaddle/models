import os
import time
import unittest
import sys
import logging

import set_env
from dataset import build_source

class TestDataSource(unittest.TestCase):
    """Test cases for dataset.source.datasource
    """
    @classmethod
    def setUpClass(cls):
        """ setup
        """
        prefix = os.path.dirname(os.path.abspath(__file__))
        cls.roi_fname = os.path.join(prefix, \
            'coco_data/COCO17_val2017.roidb')
        cls.image_dir = os.path.join(prefix, 'COCO17')
        cls.config = {'fnames': [cls.roi_fname], 'image_dir': cls.image_dir}

    @classmethod
    def tearDownClass(cls):
        """ tearDownClass """
        pass

    def test_basic(self):
        """ test basic apis 'next/size/drained'
        """
        roi_source = build_source(self.config)
        for i, sample in enumerate(roi_source): 
            self.assertTrue('image' in sample)
            self.assertGreater(len(sample['image']), 0)
        self.assertTrue(roi_source.drained())
        self.assertEqual(i + 1, roi_source.size())

    def test_reset(self):
        """ test functions 'reset/epoch_id'
        """
        roi_source = build_source(self.config)

        self.assertTrue(roi_source.next() is not None)
        self.assertEqual(roi_source.epoch_id(), 0)

        roi_source.reset()

        self.assertEqual(roi_source.epoch_id(), 1)
        self.assertTrue(roi_source.next() is not None)


if __name__ == '__main__':
    unittest.main()
