import os
import time
import unittest
import sys
import logging
import numpy as np

import set_env
from dataset import source

class TestLoader(unittest.TestCase):
    """Test cases for dataset.source.loader
    """
    @classmethod
    def setUpClass(cls):
        """ setup
        """
        cls.prefix = os.path.dirname(os.path.abspath(__file__))
        # json data
        cls.anno_path = os.path.join(cls.prefix,
            'COCO17/annotations/instances_val2017.json')
        cls.image_dir = os.path.join(cls.prefix, 'COCO17/val2017')

    @classmethod
    def tearDownClass(cls):
        """ tearDownClass """
        pass

    def test_load_coco_in_json(self):
        """ test loading COCO data in json file
        """
        anno_path = self.anno_path
        if not os.path.exists(anno_path):
            print('warning: not found %s, so skip this test' % (anno_path))
            return

        samples = 10
        records = source.load(anno_path, samples)
        self.assertEqual(len(records), samples)

        records, cname2cid = source.load(anno_path, samples, True)
        self.assertGreater(len(cname2cid), 0)

    def test_load_coco_in_roidb(self):
        """ test loading COCO data in pickled records
        """
        anno_path = os.path.join(self.prefix,
            'coco_data/instances_val2017.roidb')

        if not os.path.exists(anno_path):
            print('warning: not found %s, so skip this test' % (anno_path))
            return

        samples = 10
        records = source.load(anno_path, samples)
        self.assertEqual(len(records), samples)

        records, cname2cid = source.load(anno_path, samples, True)
        self.assertGreater(len(cname2cid), 0)

    def test_load_voc_in_xml(self):
        """ test loading VOC data in xml files
        """
        pass

    def test_load_voc_in_roidb(self):
        """ test loading VOC data in pickled records
        """
        pass


if __name__ == '__main__':
    unittest.main()

