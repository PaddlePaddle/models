import unittest
import numpy as np

import torch

import paddle
import paddle.fluid as fluid
from paddle.fluid import layers
import paddle.fluid.dygraph as dygraph
import paddle.fluid.dygraph.nn as nn

import os.path as osp
import sys

CURRENT_DIR = osp.dirname(__file__)
sys.path.append(osp.join(CURRENT_DIR, '..', '..'))

from pytracking_pp.libs.paddle_utils import n2t, t2n
import pytracking_pp.features.color as pcolor
import pytracking.features.color as tcolor


def n2p(x):
    return x


def p2n(x):
    return x


class TestColor(unittest.TestCase):
    def test_RGB(self):
        use_gpu_list = [True, False]
        for use_gpu in use_gpu_list:
            place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
            repeat = range(50)
            rng = np.random.RandomState(0)
            for _ in repeat:
                im = rng.uniform(-1, 1, (1, 3, 255, 255)).astype('float32')
                pool_stride = rng.randint(1, 5)
                if rng.uniform() < 0.5:
                    output_sz = rng.randint(120, 200, (2,))
                else:
                    output_sz = None
                normalize_power = rng.randint(1, 5)

                extractor_p = pcolor.RGB(fparams=None, pool_stride=pool_stride,
                                         output_size=output_sz,
                                         normalize_power=normalize_power,
                                         use_for_color=True, use_for_gray=True)
                extractor_t = tcolor.RGB(fparams=None, pool_stride=pool_stride,
                                         output_size=output_sz,
                                         normalize_power=normalize_power,
                                         use_for_color=True, use_for_gray=True)

                print('pool_stride: {}, output_sz: {}, normalize_power: {}'.format(
                    pool_stride, output_sz, normalize_power))
                out_p = extractor_p.get_feature(n2p(im))
                out_t = extractor_t.get_feature(n2t(im))
                out_p = p2n(out_p)
                out_t = t2n(out_t)
                np.testing.assert_allclose(out_p, out_t)

    def test_Gray(self):
        use_gpu_list = [True, False]
        for use_gpu in use_gpu_list:
            place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
            repeat = range(50)
            rng = np.random.RandomState(0)
            for _ in repeat:
                im = rng.uniform(-1, 1, (1, 3, 255, 255)).astype('float32')
                pool_stride = rng.randint(1, 5)
                if rng.uniform() < 0.5:
                    output_sz = rng.randint(120, 200, (2,))
                else:
                    output_sz = None
                normalize_power = rng.randint(1, 5)

                extractor_p = pcolor.Grayscale(fparams=None, pool_stride=pool_stride,
                                               output_size=output_sz,
                                               normalize_power=normalize_power,
                                               use_for_color=True, use_for_gray=True)
                extractor_t = tcolor.Grayscale(fparams=None, pool_stride=pool_stride,
                                               output_size=output_sz,
                                               normalize_power=normalize_power,
                                               use_for_color=True, use_for_gray=True)

                print('pool_stride: {}, output_sz: {}, normalize_power: {}'.format(
                    pool_stride, output_sz, normalize_power))
                out_p = extractor_p.get_feature(n2p(im))
                out_t = extractor_t.get_feature(n2t(im))
                out_p = p2n(out_p)
                out_t = t2n(out_t)
                np.testing.assert_allclose(out_p, out_t)


if __name__ == '__main__':
    unittest.main()
