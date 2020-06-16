#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import unittest
import numpy as np
import paddle.fluid as fluid
import pointnet_lib


def three_nn_np(x, known, eps=1e-10):
    distance = np.ones_like(x).astype('float32') * 1e40
    idx = np.zeros_like(x).astype('int32')

    b, n, _ = x.shape
    m = known.shape[1]
    for i in range(b):
        for j in range(n):
            for k in range(m):
                sub = x[i, j, :] - known[i, k, :]
                d = float(np.sum(sub * sub))
                valid_d = max(d, eps)
                if d < distance[i, j, 0]:
                    distance[i, j, 2] = distance[i, j, 1]
                    idx[i, j, 2] = idx[i, j, 1]
                    distance[i, j, 1] = distance[i, j, 0]
                    idx[i, j, 1] = idx[i, j, 0]
                    distance[i, j, 0] = valid_d
                    idx[i, j, 0] = k
                elif d < distance[i, j, 1]:
                    distance[i, j, 2] = distance[i, j, 1]
                    idx[i, j, 2] = idx[i, j, 1]
                    distance[i, j, 1] = valid_d
                    idx[i, j, 1] = k
                elif d < distance[i, j, 2]:
                    distance[i, j, 2] = valid_d
                    idx[i, j, 2] = k
    return distance, idx


class TestThreeNNOp(unittest.TestCase):
    def test_check_output(self):
        input_shape = [16, 32, 3]
        known_shape = [16, 8, 3]
        input_type = 'float32'
        eps = 1e-10

        x = fluid.data(
            name='x', shape=input_shape, dtype=input_type)
        known = fluid.data(
            name='known', shape=known_shape, dtype=input_type)
        dist, idx = pointnet_lib.three_nn(x, known, eps)

        x_np = np.random.random(input_shape).astype(input_type)
        known_np = np.random.random(known_shape).astype(input_type)
        dist_np, idx_np = three_nn_np(x_np, known_np, eps)

        place = fluid.CUDAPlace(0)
        exe = fluid.Executor(place)
        outs = exe.run(feed={'x': x_np, 'known': known_np}, fetch_list=[dist, idx])

        self.assertTrue(np.allclose(outs[0], dist_np))
        self.assertTrue(np.allclose(outs[1], idx_np))


if __name__ == "__main__":
    unittest.main()
