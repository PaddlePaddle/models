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
from utility import get_gpu_num
from .commonfunc import calculate_order_dist_matrix

class QuadrupletLoss():
    def __init__(self, 
                 train_batch_size = 80, 
                 samples_each_class = 2,
                 margin = 0.1):
        self.margin = margin
        self.samples_each_class = samples_each_class
        self.train_batch_size = train_batch_size
        num_gpus = get_gpu_num()
        assert(train_batch_size % num_gpus == 0)
        self.cal_loss_batch_size = train_batch_size // num_gpus
        assert(self.cal_loss_batch_size % samples_each_class == 0)

    def loss(self, input, label=None):
        #input = fluid.layers.l2_normalize(input, axis=1)
        input_norm = fluid.layers.sqrt(fluid.layers.reduce_sum(fluid.layers.square(input), dim=1))
        input = fluid.layers.elementwise_div(input, input_norm, axis=0)

        samples_each_class = self.samples_each_class
        batch_size = self.cal_loss_batch_size
        margin = self.margin
        d = calculate_order_dist_matrix(input, self.cal_loss_batch_size, self.samples_each_class)
        ignore, pos, neg = fluid.layers.split(d, num_or_sections= [1, 
            samples_each_class-1, batch_size-samples_each_class], dim=1)
        ignore.stop_gradient = True
        pos_max = fluid.layers.reduce_max(pos)
        neg_min = fluid.layers.reduce_min(neg)
        #pos_max = fluid.layers.sqrt(pos_max + 1e-6)
        #neg_min = fluid.layers.sqrt(neg_min + 1e-6)
        loss = fluid.layers.relu(pos_max - neg_min + margin)
        return loss
    
