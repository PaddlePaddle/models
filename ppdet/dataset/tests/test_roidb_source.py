import os
import time
import unittest
import sys
import logging

import set_env
from dataset import build_source


class TestRoiDbSource(unittest.TestCase):
    """Test cases for dataset.source.roidb_source
    """

    @classmethod
    def setUpClass(cls):
        """ setup
        """

        prefix = os.path.dirname(os.path.abspath(__file__))
        # json data
        anno_path = os.path.join(prefix, 'data/coco/instances_val2017.json')
        image_dir = os.path.join(prefix, 'data/coco/val2017')
        cls.config = {
            'fname': anno_path,
            'image_dir': image_dir,
            'samples': 100
        }

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
