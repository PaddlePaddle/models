# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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


def farthest_point_sampling_np(xyz, npoint):
    B, N, C = xyz.shape
    S = npoint

    centroids = np.zeros((B, S))
    distance = np.ones((B, N)) * 1e10
    farthest = 0
    batch_indices = np.arange(B).astype('int32')
    for i in range(S):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].reshape((B, 1, 3))
        dist = np.sum((xyz - centroid)**2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    return centroids.astype('int32')


class TestFarthestPointSamplingOp(unittest.TestCase):
    def test_check_output(self):
        x_shape = (1, 512, 3)
        x_type = 'float32'
        sampled_point_num = 256

        x = fluid.data(
            name='x', shape=x_shape, dtype=x_type)
        y = pointnet_lib.farthest_point_sampling(x, sampled_point_num)

        x_np = np.random.randint(1, 100, (x_shape[0] * x_shape[1] *
                                          3, )).reshape(x_shape).astype(x_type)
        out_np = farthest_point_sampling_np(x_np, sampled_point_num)

        place = fluid.CUDAPlace(0)
        exe = fluid.Executor(place)
        outs = exe.run(feed={'x': x_np}, fetch_list=[y])

        self.assertTrue(np.allclose(outs[0], out_np))


if __name__ == "__main__":
    unittest.main()
