import math

import torch

import unittest
import numpy as np

import paddle
import paddle.fluid as fluid
from paddle.fluid import layers
import paddle.fluid.dygraph as dygraph
import paddle.fluid.dygraph.nn as nn

import os.path as osp
import sys

CURRENT_DIR = osp.dirname(__file__)
sys.path.append(osp.join(CURRENT_DIR, '..', '..'))

from pytracking_pp.libs import TensorList as TensorListP
from pytracking.libs import TensorList as TensorListT

from pytracking_pp.libs.paddle_utils import n2p, n2t, p2n, t2n
import pytracking_pp.libs.fourier as pfo
import pytracking.libs.fourier as tfo


class TestFourier(unittest.TestCase):
    def test_rfftshift2(self):
        rng = np.random.RandomState(0)

        with fluid.dygraph.guard():
            for _ in range(10):
                input_np = rng.uniform(0, 1, [2, 3, 32, 32])

                out_p = pfo.rfftshift2(input_np)
                out_t = tfo.rfftshift2(n2t(input_np))

                out_p = out_p
                out_t = t2n(out_t)
                np.testing.assert_allclose(out_p, out_t)

    def test_irfftshift2(self):
        rng = np.random.RandomState(0)
        with fluid.dygraph.guard():
            for sz in range(3, 30):
                for _ in range(2):
                    input_np = rng.uniform(0, 1, [2, 3, sz, sz])

                    out_p = pfo.irfftshift2(input_np)
                    out_t = tfo.irfftshift2(n2t(input_np))

                    out_p = out_p
                    out_t = t2n(out_t)
                    np.testing.assert_allclose(out_p, out_t)

    def test_cfft2(self):
        rng = np.random.RandomState(0)
        with fluid.dygraph.guard():
            for sz in range(3, 30):
                for _ in range(2):
                    input_np = rng.uniform(0, 1, [2, 3, sz, sz])

                    out_p = pfo.cfft2(input_np)
                    out_t = tfo.cfft2(n2t(input_np))

                    out_p = out_p
                    out_t = t2n(out_t)
                    np.testing.assert_allclose(out_p, out_t, atol=1e-7, rtol=1e-7)

    def test_cifft2(self):
        rng = np.random.RandomState(0)

        with fluid.dygraph.guard():
            for sz in range(3, 35, 2):
                for _ in range(10):
                    input_np = rng.uniform(0, 1, [2, 3, sz, sz])

                    out_p0 = pfo.cfft2(input_np)
                    out_t0 = tfo.cfft2(n2t(input_np))

                    out_p1 = pfo.cifft2(out_p0, input_np.shape[-2:])
                    out_t1 = tfo.cifft2(out_t0, input_np.shape[-2:])

                    out_p0, out_p1 = out_p0, out_p1
                    out_t0, out_t1 = t2n(out_t0), t2n(out_t1)
                    np.testing.assert_allclose(input_np, out_p1)
                    np.testing.assert_allclose(out_p0, out_t0)
                    np.testing.assert_allclose(out_p1, out_t1)

    def test_shift_fs(self):
        rng = np.random.RandomState(0)

        with fluid.dygraph.guard():
            for sz in range(3, 35, 2):
                for _ in range(10):
                    input_np = rng.uniform(0, 1, [2, 3, sz, sz]).astype('float32')
                    shift = rng.randint(0, sz, (2,))
                    out_p0 = pfo.cfft2(input_np)
                    out_t0 = tfo.cfft2(n2t(input_np))

                    out_p1 = pfo.shift_fs(out_p0, shift)
                    out_t1 = tfo.shift_fs(out_t0, n2t(shift))

                    out_p0, out_p1 = out_p0, out_p1
                    out_t0, out_t1 = t2n(out_t0), t2n(out_t1)
                    np.testing.assert_allclose(out_p1, out_t1, atol=1e-5, rtol=1e-5)

    def test_sum_fs(self):
        rng = np.random.RandomState(0)

        with fluid.dygraph.guard():
            for sz in range(3, 35, 2):
                for _ in range(10):
                    input_np0 = rng.uniform(0, 1, [2, 3, sz, sz]).astype('float32')
                    input_np1 = rng.uniform(0, 1, [2, 3, sz, sz]).astype('float32')
                    shift0 = math.pi * (1 - rng.randint(0, sz, (2,)) / sz)
                    shift1 = math.pi * (1 - rng.randint(0, sz, (2,)) / sz)

                    input_p = TensorListP([input_np0, input_np1])
                    input_t = TensorListT([input_np0, input_np1]).apply(n2t)
                    shift_p = TensorListP([shift0, shift1])
                    shift_t = TensorListT([shift0, shift1]).apply(n2t)

                    out_p0 = pfo.cfft2(input_p)
                    out_t0 = tfo.cfft2(input_t)

                    out_p1 = pfo.shift_fs(out_p0, shift_p)
                    out_t1 = tfo.shift_fs(out_t0, shift_t)

                    out_p2 = pfo.sum_fs(out_p1)
                    out_t2 = tfo.sum_fs(out_t1)

                    out_p0, out_p1, out_p2 = out_p0, out_p1, out_p2
                    out_t0, out_t1, out_t2 = out_t0.apply(t2n), out_t1.apply(t2n), t2n(out_t2)
                    np.testing.assert_allclose(out_p2, out_t2, atol=1e-5, rtol=1e-5)


if __name__ == '__main__':
    unittest.main()
