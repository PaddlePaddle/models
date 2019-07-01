import os
import time
import unittest
import sys
import logging
import numpy as np

import set_env
from data import build_source
from data import transform as tf

logger = logging.getLogger(__name__)

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
        anno_path = set_env.coco_data['TRAIN']['ANNO_FILE']
        image_dir = set_env.coco_data['TRAIN']['IMAGE_DIR']
        cls.sc_config = {
            'anno_file': anno_path,
            'image_dir': image_dir,
            'samples': 200
        }

        cls.ops = [{
            'op': 'DecodeImage',
            'to_rgb': True
        }, {
            'op': 'ResizeImage',
            'target_size': 800,
            'max_size': 1333
        }, {
            'op': 'ArrangeRCNN',
            'is_mask': False
        }]

    @classmethod
    def tearDownClass(cls):
        """ tearDownClass """
        pass

    def test_map(self):
        """ test transformer.map
        """
        mapper = tf.build(self.ops)
        ds = build_source(self.sc_config)
        mapped_ds = tf.map(ds, mapper)
        ct = 0
        for sample in mapped_ds:
            self.assertTrue(type(sample[0]) is np.ndarray)
            ct += 1

        self.assertEqual(ct, mapped_ds.size())

    def test_parallel_map(self):
        """ test transformer.map with concurrent workers
        """
        mapper = tf.build(self.ops)
        ds = build_source(self.sc_config)
        worker_conf = {'WORKER_NUM': 2, 'use_process': True}
        mapped_ds = tf.map(ds, mapper, worker_conf)

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
        mapper = tf.build(self.ops)
        ds = build_source(self.sc_config)
        mapped_ds = tf.map(ds, mapper)
        batched_ds = tf.batch(mapped_ds, batchsize, True)
        for sample in batched_ds:
            out = sample
        self.assertEqual(len(out), batchsize)


if __name__ == '__main__':
    unittest.main()
