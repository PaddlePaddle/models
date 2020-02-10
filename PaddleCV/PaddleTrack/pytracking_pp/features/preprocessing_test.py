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

from pytracking_pp.libs.paddle_utils import n2p, n2t, p2n, t2n
import pytracking_pp.features.preprocessing as ppre
import pytracking.features.preprocessing as tpre

def test_biliinear():
    from paddle import fluid
    from paddle.fluid import layers, dygraph
    import numpy as np
    def n2p(x):
        return dygraph.to_variable(np.array(x))

    with fluid.dygraph.guard():
        x = np.random.rand(1, 3, 100, 100)
        y = layers.resize_bilinear(n2p(x), [200, 200])

class TestPreprocess(unittest.TestCase):
    def test_sample_patch(self):
        repeat = range(50)
        rng = np.random.RandomState(0)
        for _ in repeat:
            im = rng.uniform(-1, 1, (1, 3, 255, 255))
            pos = rng.randint(0, 255, (2,))
            sample_sz = rng.randint(10, 127, (2,))
            output_sz = rng.randint(120, 200, (2,))
            print('test pos: {}, sample_sz: {}, output_sz: {}'.format(pos, sample_sz, output_sz))
            im_p = np.transpose(im[0], (1, 2, 0))
            out_p = ppre.sample_patch(im_p, pos, sample_sz, output_sz)
            out_t = tpre.sample_patch(n2t(im), n2t(pos), n2t(sample_sz), n2t(output_sz))

            out_p = np.transpose(np.expand_dims(out_p, 0), (0, 3, 1, 2))
            out_t = t2n(out_t)
            np.testing.assert_allclose(out_p, out_t, atol=1e-5, rtol=1e-5)

if __name__ == '__main__':
    unittest.main()
