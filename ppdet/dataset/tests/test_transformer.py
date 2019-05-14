import os
import time
import unittest
import sys
import logging
import numpy as np

import set_env
from dataset import build_source
from dataset.transform import operator as op
from dataset.transform import transformer

logging.basicConfig(level=logging.INFO)
class TestTransformer(unittest.TestCase):
    """Test cases for dataset.transform.transformer
    """
    @classmethod
    def setUpClass(cls):
        """ setup
        """
        prefix = os.path.dirname(os.path.abspath(__file__))
        # json data
        anno_path = os.path.join(prefix,
            'COCO17/annotations/instances_val2017.json')
        image_dir = os.path.join(prefix, 'COCO17/val2017')
        cls.sc_config = {'fname': anno_path,
            'image_dir': image_dir, 'samples': 200}
        
        cls.ops = [op.DecodeImage(to_rgb=True),
            op.RandFlipImage(prob=0.0),
            op.ResizeImage(target_size=300, max_size=1333),
            op.NormalizeImage(mean=[108, 108, 108]),
            op.Bgr2Rgb(),
            op.ArrangeSample(is_mask=False)]

    @classmethod
    def tearDownClass(cls):
        """ tearDownClass """
        pass

    def test_map(self):
        """ test transformer.map
        """
        mapper = op.build(self.ops)
        ds = build_source(self.sc_config)
        mapped_ds = transformer.map(ds, mapper)
        ct = 0
        for sample in mapped_ds:
            self.assertTrue(type(sample[0]) is np.ndarray)
            ct += 1

        self.assertEqual(ct, mapped_ds.size())

    def test_thread_map(self):
        """ test transformer.map with concurrent workers
        """
        mapper = op.build(self.ops)
        ds = build_source(self.sc_config)
        worker_conf = {'worker_num': 2}
        mapped_ds = transformer.map(
            ds, mapper, worker_conf)

        ct = 0
        for sample in mapped_ds:
            self.assertTrue(type(sample[0]) is np.ndarray)
            ct += 1

        self.assertTrue(mapped_ds.drained())
        self.assertEqual(ct, mapped_ds.size())
        mapped_ds.reset()

        ct = 0
        for sample in mapped_ds:
            self.assertTrue(type(sample[0]) is np.ndarray)
            ct += 1

        self.assertEqual(ct, mapped_ds.size())

    def test_batch(self):
        """ test batched dataset
        """
        batchsize = 2
        ds = build_source(self.sc_config)
        batched_ds = transformer.batch(ds, batchsize)

        for sample in batched_ds:
            self.assertEqual(len(sample), batchsize)

        batched_ds.reset()


if __name__ == '__main__':
    unittest.main()
