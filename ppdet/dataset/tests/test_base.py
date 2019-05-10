import os
import unittest
import logging
import numpy as np

import set_env
from dataset.transform import operator

logging.basicConfig(level=logging.INFO)


class TestBase(unittest.TestCase):
    """Test cases for dataset.transform.operator
    """
    @classmethod
    def setUpClass(cls):
        """ setup
        """
        roidb_root = '/home/data/VOC.roidb'
        import pickle as pkl
        with open(roidb_root, 'rb') as f:
            roidb = f.read()
            roidb = pkl.loads(roidb)
        fn = os.path.join('/home/VOC/JPEGImages', roidb[0]['image_url'])
        with open(fn, 'rb') as f:
            roidb[0]['image'] = f.read()
        cls.sample = roidb[0]

    @classmethod
    def tearDownClass(cls):
        """ tearDownClass """
        pass

    def test_ops(self):
        """ test operators
        """
        # ResizeImage
        ops_conf = [{'name': 'DecodeImage'},
                    {'name': 'ResizeImage',
                     'params': {'target_size': 300, 'max_size': 1333}}]
        mapper = operator.build(ops_conf)
        self.assertTrue(mapper is not None)
        result0 = mapper(self.sample)
        self.assertIsNotNone(result0['image'])
        self.assertEqual(len(result0['image'].shape), 3)
        # RandFlipImage
        ops_conf = [{'name': 'RandFlipImage'}]
        mapper = operator.build(ops_conf)
        self.assertTrue(mapper is not None)
        result1 = mapper(result0)
        self.assertEqual(result1['image'].shape, result0['image'].shape)
        self.assertEqual(result1['gt_bbox'].shape, result0['gt_bbox'].shape)
        # NormalizeImage
        ops_conf = [{'name': 'NormalizeImage'}]
        mapper = operator.build(ops_conf)
        self.assertTrue(mapper is not None)
        result2 = mapper(result1)
        im1 = result1['image']
        count = np.where(im1 <= 1)[0]
        if im1.dtype == 'float64':
            self.assertEqual(count, im1.shape[0]*im1.shape[1], im1.shape[2])
        # ArrangeSample
        ops_conf = [{'name': 'ArrangeSample'}]
        mapper = operator.build(ops_conf)
        self.assertTrue(mapper is not None)
        result3 = mapper(result2)
        self.assertEqual(type(result3), tuple)


if __name__ == '__main__':
    unittest.main()
