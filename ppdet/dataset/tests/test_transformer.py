import os
import time
import unittest
import sys
import logging
import numpy as np

import set_env
from dataset import build_source
from dataset.transform import operator as op
from dataset.transform import transform

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
                        #op.RandFlipImage(prob=0.0), # there is a bug here
                        op.ResizeImage(target_size=300, max_size=1333),
                        op.NormalizeImage(mean=[108, 108, 108]),
                        op.Bgr2Rgb(),
                        op.ArrangeSample(is_mask=False)]

    @classmethod
    def tearDownClass(cls):
        """ tearDownClass """
        pass

    def test_op(self):
        """ test base operator
        """
        ops = [op.BaseOperator()]
        mapper = op.build(ops)
        self.assertTrue(mapper is not None)

        fake_sample = {'image': 'image data', 'label': 1234}
        result = mapper(fake_sample)
        self.assertTrue(result is not None)

    def test_transform(self):
        """ test transformed dataset
        """
        sc = build_source(self.sc_config)
        mapper = op.build(self.ops)
        result_data = transform(sc, self.ops)

        for sample in result_data:
            self.assertTrue(type(sample[0]) is np.ndarray)

    def test_fast_transform(self):
        """ test fast transformer
        """
        sc = build_source(self.sc_config)
        mapper = op.build(self.ops)
        worker_conf = {'worker_num': 2}
        result_data = transform(sc,
            self.ops, worker_conf)

        ct = 0
        for sample in result_data:
            self.assertTrue(type(sample[0]) is np.ndarray)
            ct += 1

        self.assertTrue(result_data.drained())
        self.assertEqual(ct, result_data.size())
        result_data.reset()

        ct = 0
        for sample in result_data:
            self.assertTrue(type(sample[0]) is np.ndarray)
            ct += 1

        self.assertEqual(ct, result_data.size())


if __name__ == '__main__':
    unittest.main()
