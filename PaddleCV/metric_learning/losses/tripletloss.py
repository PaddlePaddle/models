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

import paddle.fluid as fluid

class TripletLoss():
    def __init__(self, margin=0.1):
        self.margin = margin

    def loss(self, input, label=None):
        margin = self.margin
        fea_dim = input.shape[1] # number of channels
        #input = fluid.layers.l2_normalize(input, axis=1)
        input_norm = fluid.layers.sqrt(fluid.layers.reduce_sum(fluid.layers.square(input), dim=1))
        input = fluid.layers.elementwise_div(input, input_norm, axis=0)
        output = fluid.layers.reshape(input, shape = [-1, 3, fea_dim])

        anchor, positive, negative = fluid.layers.split(output, num_or_sections = 3, dim = 1)
 
        anchor = fluid.layers.reshape(anchor, shape = [-1, fea_dim])
        positive = fluid.layers.reshape(positive, shape = [-1, fea_dim])
        negative = fluid.layers.reshape(negative, shape = [-1, fea_dim])
 
        a_p = fluid.layers.square(anchor - positive)
        a_n = fluid.layers.square(anchor - negative)
        a_p = fluid.layers.reduce_sum(a_p, dim = 1)
        a_n = fluid.layers.reduce_sum(a_n, dim = 1)
        #a_p = fluid.layers.sqrt(a_p + 1e-6)
        #a_n = fluid.layers.sqrt(a_n + 1e-6)
        loss = fluid.layers.relu(a_p + margin - a_n)
        return loss
