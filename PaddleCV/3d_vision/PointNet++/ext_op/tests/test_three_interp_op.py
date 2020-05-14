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


def three_interp_np(x, weight, idx):
    b, m, c = x.shape
    n = weight.shape[1]

    output = np.zeros((b, n, c)).astype('float32')
    for i in range(b):
        for j in range(n):
            w1, w2, w3 = weight[i, j, :]
            i1, i2, i3 = idx[i, j, :]
            output[i, j, :] = w1 * x[i, i1, :] \
                            + w2 * x[i, i2, :] \
                            + w3 * x[i, i3, :]
    return output


class TestThreeInterpOp(unittest.TestCase):
    def test_check_output(self):
        input_shape = [8, 21, 29]
        input_type = 'float32'
        weight_shape = [8, 37, 3]
        weight_type = 'float32'

        x = fluid.data(
            name='x', shape=input_shape, dtype=input_type)
        weight = fluid.data(
            name='weight', shape=weight_shape, dtype=weight_type)
        idx = fluid.data(
            name='idx', shape=weight_shape, dtype="int32")
        y = pointnet_lib.three_interp(x, weight, idx)

        x_np = np.random.random(input_shape).astype(input_type)
        weight_np = np.random.random(weight_shape).astype(weight_type)
        idx_np = np.random.uniform(0, input_shape[1], weight_shape).astype("int32")
        out_np = three_interp_np(x_np, weight_np, idx_np)

        place = fluid.CUDAPlace(0)
        exe = fluid.Executor(place)
        outs = exe.run(feed={'x': x_np, 'weight': weight_np, 'idx': idx_np}, fetch_list=[y])

        self.assertTrue(np.allclose(outs[0], out_np))


if __name__ == "__main__":
    unittest.main()
