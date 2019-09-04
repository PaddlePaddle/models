#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import paddle as paddle
import paddle.fluid as fluid

def generate_index(batch_size, samples_each_class):
    a = np.arange(0, batch_size * batch_size) # N*N x 1
    a = a.reshape(-1, batch_size) # N x N
    steps = batch_size // samples_each_class
    res = []
    for i in range(batch_size):
        step = i // samples_each_class
        start = step * samples_each_class
        end = (step + 1) * samples_each_class
        p = []
        n = []
        for j, k in enumerate(a[i]):
            if j >= start and j < end:
                if j == i:
                    p.insert(0, k)
                else:
                    p.append(k)
            else:
                n.append(k)
        comb = p + n
        res += comb
    res = np.array(res).astype(np.int32)
    return res

def calculate_order_dist_matrix(feature, batch_size, samples_each_class):
    assert(batch_size % samples_each_class == 0)
    feature = fluid.layers.reshape(feature, shape=[batch_size, -1])
    ab = fluid.layers.matmul(feature, feature, False, True)
    a2 = fluid.layers.square(feature)
    a2 = fluid.layers.reduce_sum(a2, dim = 1)
    d = fluid.layers.elementwise_add(-2*ab, a2, axis = 0)
    d = fluid.layers.elementwise_add(d, a2, axis = 1)
    d = fluid.layers.reshape(d, shape = [-1, 1])
    index = generate_index(batch_size, samples_each_class)
    index_var = fluid.layers.create_global_var(shape=[batch_size*batch_size], value=0, dtype='int32', persistable=True)
    index_var = fluid.layers.assign(index, index_var)
    d = fluid.layers.gather(d, index=index_var)
    d = fluid.layers.reshape(d, shape=[-1, batch_size])
    return d
