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


def query_ball_point_np(points, new_points, radius, nsample):
    b, n, c = points.shape
    _, m, _ = new_points.shape
    out = np.zeros(shape=(b, m, nsample)).astype('int32')
    radius_2 = radius * radius
    for i in range(b):
        for j in range(m):
            cnt = 0
            for k in range(n):
                if (cnt == nsample):
                    break
                dist = np.sum(np.square(points[i][k] - new_points[i][j]))
                if (dist < radius_2):
                    if cnt == 0:
                        out[i][j] = np.ones(shape=(nsample)) * k
                    out[i][j][cnt] = k
                    cnt += 1
    return out


class TestQueryBallOp(unittest.TestCase):
    def test_check_output(self):
        points_shape = [2, 5, 3]
        new_points_shape = [2, 4, 3]
        points_type = 'float32'
        radius = 6
        nsample = 5

        points = fluid.data(
            name='points', shape=points_shape, dtype=points_type)
        new_points = fluid.data(
            name='new_points', shape=new_points_shape, dtype=points_type)
        y = pointnet_lib.query_ball(points, new_points, radius, nsample)

        points_np = np.random.randint(1, 5, points_shape).astype(points_type)
        new_points_np = np.random.randint(1, 5, new_points_shape).astype(points_type)
        out_np = query_ball_point_np(points_np, new_points_np, radius, nsample)

        place = fluid.CUDAPlace(0)
        exe = fluid.Executor(place)
        outs = exe.run(feed={'points': points_np, 'new_points': new_points_np}, fetch_list=[y])

        self.assertTrue(np.allclose(outs[0], out_np))


if __name__ == "__main__":
    unittest.main()
