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


def group_points_np(x, idx):
    b, m, s = idx.shape
    _, c, n = x.shape

    output = np.zeros((b, c, m, s)).astype(x.dtype)
    for i in range(b):
        for j in range(m):
            for k in range(s):
                output[i, :, j, k] = x[i, :, idx[i, j, k]]
    return output


class TestGroupPointsOp(unittest.TestCase):
    def test_check_output(self):
        x_shape = [8, 43, 29]
        x_type = 'float32'
        idx_shape = [8, 37, 41]
        idx_type = 'int32'

        x = fluid.layers.data(
            name='x', shape=x_shape, dtype=x_type, append_batch_size=False)
        idx = fluid.layers.data(
            name='idx', shape=idx_shape, dtype=idx_type, append_batch_size=False)
        y = pointnet_lib.group_points(x, idx)

        x_np = np.random.uniform(-10, 10, x_shape).astype(x_type)
        idx_np = np.random.randint(0, x_shape[2], idx_shape).astype(idx_type)
        out_np = group_points_np(x_np, idx_np)

        place = fluid.CUDAPlace(0)
        exe = fluid.Executor(place)
        outs = exe.run(feed={'x': x_np, 'idx': idx_np}, fetch_list=[y])

        self.assertTrue(np.allclose(outs[0], out_np))


if __name__ == "__main__":
    unittest.main()
