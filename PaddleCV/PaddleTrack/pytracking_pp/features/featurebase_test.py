import numpy as np
import torch
from paddle import fluid
from paddle.fluid import layers

import os.path as osp
import sys

CURRENT_DIR = osp.dirname(__file__)
sys.path.append(osp.join(CURRENT_DIR, '..', '..'))

from pytracking_pp.libs.paddle_utils import n2p, n2t, p2n, t2n, broadcast_op
from bilib import crash_on_ipy


def test_normalize():
    rng = np.random.RandomState(0)
    feat_np = rng.uniform(-1, 1, (10, 64, 18, 18)).astype('float32')
    for norm in [2, 3, 4]:
        # paddle normalize
        with fluid.dygraph.guard():
            feat = n2p(feat_np)
            feat1_p = (layers.reduce_sum(
                layers.reshape(layers.abs(feat), [feat.shape[0], 1, 1, -1]) ** norm,
                dim=3,
                keep_dim=True) /
                       (feat.shape[1] * feat.shape[2] * feat.shape[3]) + 1e-10) ** (1 / norm)
            feat2_p = broadcast_op(feat, feat1_p, 'div')

            # torch normalize
            feat = n2t(feat_np)
            feat1_t = (torch.sum(feat.abs().view(feat.shape[0], 1, 1, -1) ** norm, dim=3, keepdim=True) /
                       (feat.shape[1] * feat.shape[2] * feat.shape[3]) + 1e-10) ** (1 / norm)
            feat2_t = feat / feat1_t

            np.testing.assert_allclose(p2n(feat1_p), t2n(feat1_t), atol=1e-5, rtol=1e-5)
            np.testing.assert_allclose(p2n(feat2_p), t2n(feat2_t), atol=1e-5, rtol=1e-5)


if __name__ == '__main__':
    test_normalize()
