import os
import time
import unittest
import sys
import logging
import numpy as np

import set_env
from dataset import build_source
from dataset.transform import operator
from dataset.transform import transform

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
        cls.sc_config = {'fnames': [anno_path],
            'image_dir': image_dir, 'samples': 100}

        cls.ops_conf = [{'name': 'DecodeImage', \
            'params': {'to_rgb': True}}]

    @classmethod
    def tearDownClass(cls):
        """ tearDownClass """
        pass

    def test_op(self):
        """ test base operator
        """
        ops_conf = [{'name': 'BaseOperator'}]
        mapper = operator.build(ops_conf)
        self.assertTrue(mapper is not None)

        fake_sample = {'image': 'image data', 'label': 1234}
        result = mapper(fake_sample)
        self.assertTrue(result is not None)

    def test_transform(self):
        """ test transformed dataset
        """
        sc = build_source(self.sc_config)
        mapper = operator.build(self.ops_conf)
        worker_conf = {}
        result_data = transform(sc, \
            self.ops_conf, worker_conf)

        for sample in result_data:
            self.assertTrue(type(sample['image']) is np.ndarray)


if __name__ == '__main__':
    unittest.main()

