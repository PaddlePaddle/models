import unittest
import numpy as np

import torch
from torch.nn.functional import conv2d as Tconv2d

import paddle
import paddle.fluid as fluid
from paddle.fluid import layers
import paddle.fluid.dygraph as dygraph
import paddle.fluid.dygraph.nn as nn

import os.path as osp
import sys

CURRENT_DIR = osp.dirname(__file__)
sys.path.append(osp.join(CURRENT_DIR, '..', '..'))

from pytracking_pp.libs.Fconv2d import Fconv2d as Pconv2d
from pytracking_pp.libs.paddle_utils import n2p, n2t, p2n, t2n

def test_conv2d():
    import numpy as np
    from paddle.fluid import dygraph
    from pytracking_pp.libs.Fconv2d import Fconv2d as Pconv2d

    def n2p(x): return dygraph.to_variable(np.array(x))

    rng = np.random.RandomState(0)
    with dygraph.guard():
        input_np = rng.uniform(0, 1, [2, 3, 32, 32]).astype('float32')
        filter_np = rng.uniform(0, 1, [6, 3, 3, 3]).astype('float32')

        input_p = n2p(input_np)
        filter_p = n2p(filter_np)
        out_p = Pconv2d(input_p, filter_p)


class TestFconv2d(unittest.TestCase):
    def test_forward(self):
        rng = np.random.RandomState(0)
        with dygraph.guard():
            input_np = rng.uniform(0, 1, [2, 3, 32, 32]).astype('float32')
            filter_np = rng.uniform(0, 1, [6, 3, 3, 3]).astype('float32')

            input_p = n2p(input_np)
            filter_p = n2p(filter_np)
            out_p = p2n(Pconv2d(input_p, filter_p))

            input_t = n2t(input_np)
            filter_t = n2t(filter_np)
            out_t = t2n(Tconv2d(input_t, filter_t))

            np.testing.assert_allclose(out_p, out_t)


if __name__ == '__main__':
    unittest.main()
