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

import math
import paddle.fluid as fluid
from utility import get_gpu_num
from .commonfunc import calculate_order_dist_matrix

class EmlLoss():
    def __init__(self, train_batch_size = 40, samples_each_class=2):
        self.samples_each_class = samples_each_class
        self.train_batch_size = train_batch_size
        num_gpus = get_gpu_num()
        assert(train_batch_size % num_gpus == 0)
        self.cal_loss_batch_size = train_batch_size // num_gpus
        assert(self.cal_loss_batch_size % samples_each_class == 0)

    def surrogate_function(self, beta, theta, bias):
        x = theta * fluid.layers.exp(bias) 
        output = fluid.layers.log(1+beta*x)/math.log(1+beta)
        return output

    def surrogate_function_approximate(self, beta, theta, bias):
        output = (fluid.layers.log(theta) + bias + math.log(beta))/math.log(1+beta)
        return output

    def surrogate_function_stable(self, beta, theta, target, thresh):
        max_gap = fluid.layers.fill_constant([1], dtype='float32', value=thresh)
        max_gap.stop_gradient = True

        target_max = fluid.layers.elementwise_max(target, max_gap)
        target_min = fluid.layers.elementwise_min(target, max_gap)
        
        loss1 = self.surrogate_function(beta, theta, target_min)
        loss2 = self.surrogate_function_approximate(beta, theta, target_max)
        bias = self.surrogate_function(beta, theta, max_gap)
        loss = loss1 + loss2 - bias
        return loss

    def loss(self, input, label=None):
        samples_each_class = self.samples_each_class
        batch_size = self.cal_loss_batch_size
        #input = fluid.layers.l2_normalize(input, axis=1)
        #input_norm = fluid.layers.sqrt(fluid.layers.reduce_sum(fluid.layers.square(input), dim=1))
        #input = fluid.layers.elementwise_div(input, input_norm, axis=0)
        d = calculate_order_dist_matrix(input, self.cal_loss_batch_size, self.samples_each_class)
        ignore, pos, neg = fluid.layers.split(d, num_or_sections= [1, 
            samples_each_class-1, batch_size-samples_each_class], dim=1)
        ignore.stop_gradient = True 
                    
        pos_max = fluid.layers.reduce_max(pos, dim=1)
        pos_max = fluid.layers.reshape(pos_max, shape=[-1, 1])
        pos = fluid.layers.exp(pos - pos_max)
        pos_mean = fluid.layers.reduce_mean(pos, dim=1)

        neg_min = fluid.layers.reduce_min(neg, dim=1)
        neg_min = fluid.layers.reshape(neg_min, shape=[-1, 1])
        neg = fluid.layers.exp(-1*(neg-neg_min))
        neg_mean = fluid.layers.reduce_mean(neg, dim=1)
        bias = pos_max - neg_min
        theta = fluid.layers.reshape(neg_mean * pos_mean, shape=[-1,1])
        thresh = 20.0
        beta = 100000
        loss = self.surrogate_function_stable(beta, theta, bias, thresh)
        return loss
