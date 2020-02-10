import unittest
import numpy as np

import torch

import os.path as osp
import sys

CURRENT_DIR = osp.dirname(__file__)
sys.path.append(osp.join(CURRENT_DIR, '..', '..'))

from pytracking_pp.libs.paddle_utils import n2p, n2t, p2n, t2n
import pytracking_pp.libs.dcf as pdcf
import pytracking.libs.dcf as tdcf


def test_elementwisemul():
    import numpy as np
    from paddle import fluid
    from paddle.fluid import layers, dygraph

    def n2p(x): return dygraph.to_variable(np.array(x))

    with dygraph.guard(fluid.CPUPlace()):
        x = np.random.uniform(-1, 1, (1, 1, 288, 288))
        y = np.random.uniform(-1, 1, (1, 1, 288, 288))
        x_p, y_p = n2p(x), n2p(y)
        out_p = layers.elementwise_mul(x_p, y_p)


class TestDCF(unittest.TestCase):
    def test_cos(self):
        rng = np.random.RandomState(0)
        input_np = rng.uniform(0, 1, [2, 3, 32, 32])

        out_p = np.cos(input_np)
        out_t = torch.cos(n2t(input_np))

        out_t = t2n(out_t)
        np.testing.assert_allclose(out_p, out_t, atol=1e-6)

    def test_hann1d(self):
        """Note: rtol = 1e-6 will fail the tests"""
        centered = [True, False]
        sizes = range(3, 30)
        for c in centered:
            for s in sizes:
                out_p = pdcf.hann1d(s, c)
                out_t = tdcf.hann1d(s, c)

                out_t = t2n(out_t)
                np.testing.assert_allclose(out_p, out_t, rtol=1e-5)

    def test_hann2d(self):
        centered = [True, False]
        sizes = range(3, 30)
        for c in centered:
            for s in sizes:
                sz_np = np.array([s, s], dtype='int')
                out_p = pdcf.hann2d(sz_np, c)
                out_t = tdcf.hann2d(n2t(sz_np), c)

                out_t = t2n(out_t)
                np.testing.assert_allclose(out_p, out_t, rtol=1e-5)

    def test_hann2d_clipped(self):
        centered = [True, False]
        sizes = range(3, 30)
        enlarge = range(2, 10, 2)
        for c in centered:
            for s in sizes:
                for en in enlarge:
                    sz_np = np.array([s, s], dtype='int')
                    out_p = pdcf.hann2d_clipped(sz_np + en, sz_np, c)
                    out_t = tdcf.hann2d_clipped(n2t(sz_np + en), n2t(sz_np), c)

                    out_t = t2n(out_t)
                    np.testing.assert_allclose(out_p, out_t, rtol=1e-5)

    def test_gauss_fourier(self):
        halfed = [True, False]
        sizes = range(3, 30)
        repeat = range(10)
        rng = np.random.RandomState(0)
        for h in halfed:
            for s in sizes:
                for _ in repeat:
                    sigma = rng.uniform(0, 10)
                    out_p = pdcf.gauss_fourier(s, sigma, h)
                    out_t = tdcf.gauss_fourier(s, sigma, h)

                    out_t = t2n(out_t)
                    np.testing.assert_allclose(out_p, out_t, rtol=1e-5, atol=1e-6)

    def test_gauss_spatial(self):
        sizes = range(3, 30)
        repeat = range(10)
        rng = np.random.RandomState(0)
        for s in sizes:
            for _ in repeat:
                sigma = rng.uniform(0, 10)
                center = rng.uniform(-10, 10)
                end_pad = rng.randint(0, 10)
                out_p = pdcf.gauss_spatial(s, sigma, center, end_pad)
                out_t = tdcf.gauss_spatial(s, sigma, center, end_pad)

                out_t = t2n(out_t)
                np.testing.assert_allclose(out_p, out_t, rtol=1e-5, atol=1e-6)

    def test_label_function(self):
        repeat = range(50)
        rng = np.random.RandomState(0)
        for _ in repeat:
            sigma = rng.uniform(0.001, 10, (2,))
            sz = rng.randint(1, 10, (2,))
            out_p = pdcf.label_function(sz, sigma)
            out_t = tdcf.label_function(n2t(sz), n2t(sigma))

            out_t = t2n(out_t)
            np.testing.assert_allclose(out_p, out_t, rtol=1e-5, atol=1e-6)

    def test_label_function_spatial(self):
        repeat = range(100)
        rng = np.random.RandomState(0)
        for _ in repeat:
            sigma = rng.uniform(0.001, 10, (2,)).astype(np.float32)
            sz = rng.randint(1, 10, (2,))
            center = rng.uniform(-10, 10, (2,)).astype(np.float32)
            end_pad = rng.randint(1, 10, (2,)).astype(np.float32)
            out_p = pdcf.label_function_spatial(sz, sigma, center, end_pad)
            out_t = tdcf.label_function_spatial(n2t(sz), n2t(sigma), n2t(center), n2t(end_pad))

            out_t = t2n(out_t)
            np.testing.assert_allclose(out_p, out_t, rtol=1e-5, atol=1e-6)

    def test_max2d(self):
        repeat = range(100)
        rng = np.random.RandomState(0)
        for _ in repeat:
            response = rng.uniform(0.001, 10, (4, 5, 100, 200)).astype(np.float32)
            out_p1, out_p2 = pdcf.max2d(response)
            out_t1, out_t2 = tdcf.max2d(n2t(response))

            out_t1, out_t2 = t2n(out_t1), t2n(out_t2)
            np.testing.assert_allclose(out_p1, out_t1, rtol=1e-5, atol=1e-6)
            np.testing.assert_allclose(out_p2, out_t2, rtol=1e-5, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
