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

class NpairsLoss():
    def __init__(self, 
                 train_batch_size = 160, 
                 samples_each_class=2, 
                 reg_lambda=0.01):
        self.samples_each_class = samples_each_class
        assert(self.samples_each_class == 2)
        self.train_batch_size = train_batch_size
        num_gpus = get_gpu_num()
        assert(train_batch_size % num_gpus == 0)
        self.cal_loss_batch_size = train_batch_size // num_gpus
        assert(self.cal_loss_batch_size % samples_each_class == 0)
        self.reg_lambda = reg_lambda

    def loss(self, input, label=None):
        reg_lambda = self.reg_lambda
        samples_each_class = self.samples_each_class
        batch_size = self.cal_loss_batch_size
        num_class = batch_size // samples_each_class
        fea_dim = input.shape[1]
       
        input = fluid.layers.reshape(input, shape = [-1, fea_dim])
        feature = fluid.layers.reshape(input, shape = [-1, samples_each_class, fea_dim])
        label = fluid.layers.reshape(label, shape = [-1, samples_each_class])
        label = fluid.layers.cast(label, dtype='float32')
        if samples_each_class == 2:
            anchor_fea, positive_fea = fluid.layers.split(feature, num_or_sections = 2, dim = 1)
            anchor_lab, positive_lab = fluid.layers.split(label, num_or_sections = 2, dim = 1)
        else:
            anchor_fea, positive_fea = fluid.layers.split(feature, num_or_sections = [1, samples_each_class-1], dim = 1)
            anchor_lab, positive_lab = fluid.layers.split(label, num_or_sections = [1, samples_each_class-1], dim = 1)

        anchor_fea = fluid.layers.reshape(anchor_fea, shape = [-1, fea_dim])
        positive_fea = fluid.layers.reshape(positive_fea, shape = [-1, fea_dim])
        positive_fea_trans = fluid.layers.transpose(positive_fea, perm = [1, 0])
        similarity_matrix = fluid.layers.mul(anchor_fea, positive_fea_trans)

        anchor_lab = fluid.layers.expand(x=anchor_lab, expand_times=[1, batch_size-num_class])
        positive_lab_tran = fluid.layers.transpose(positive_lab, perm = [1, 0])
        positive_lab_tran = fluid.layers.expand(x=positive_lab_tran, expand_times=[num_class, 1])
        label_remapped = fluid.layers.equal(anchor_lab, positive_lab_tran)
        label_remapped = fluid.layers.cast(label_remapped, dtype='float32') / (samples_each_class-1)
        label_remapped.stop_gradient = True

        out = fluid.layers.softmax(input=similarity_matrix, use_cudnn=False)
        xentloss = fluid.layers.cross_entropy(input=out, label=label_remapped, soft_label=True)
        xentloss = fluid.layers.mean(x=xentloss)

        reg = fluid.layers.reduce_mean(fluid.layers.reduce_sum(fluid.layers.square(input), dim=1))
        l2loss = 0.5 * reg_lambda * reg
        return xentloss + l2loss
