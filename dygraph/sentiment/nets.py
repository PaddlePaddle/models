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
import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, FC, Embedding
from paddle.fluid.dygraph.base import to_variable


class SimpleConvPool(fluid.dygraph.Layer):
    def __init__(self,
                 name_scope,
                 num_filters,
                 filter_size,
                 use_cudnn=False,
                 batch_size=None):
        super(SimpleConvPool, self).__init__(name_scope)
        self.batch_size = batch_size
        self._conv2d = Conv2D(
            self.full_name(),
            num_filters=num_filters,
            filter_size=filter_size,
            padding=[1, 1],
            use_cudnn=use_cudnn,
            act='tanh')

    def forward(self, inputs):
        x = self._conv2d(inputs)
        x = fluid.layers.reduce_max(x, dim=-1)
        x = fluid.layers.reshape(x, shape=[self.batch_size, -1])
        return x


class CNN(fluid.dygraph.Layer):
    def __init__(self, name_scope, dict_dim, batch_size, seq_len):
        super(CNN, self).__init__(name_scope)
        self.dict_dim = dict_dim
        self.emb_dim = 128
        self.hid_dim = 128
        self.fc_hid_dim = 96
        self.class_dim = 2
        self.win_size = [3, self.hid_dim]
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.embedding = Embedding(
            self.full_name(),
            size=[self.dict_dim + 1, self.emb_dim],
            dtype='float32',
            is_sparse=False)

        self._simple_conv_pool_1 = SimpleConvPool(
            self.full_name(),
            self.hid_dim,
            self.win_size,
            batch_size=self.batch_size)
        self._fc1 = FC(self.full_name(), size=self.fc_hid_dim, act="softmax")
        self._fc_prediction = FC(self.full_name(),
                                 size=self.class_dim,
                                 act="softmax")

    def forward(self, inputs, label=None):
        emb = self.embedding(inputs)
        o_np_mask = (inputs.numpy() != self.dict_dim).astype('float32')
        mask_emb = fluid.layers.expand(
            to_variable(o_np_mask), [1, self.hid_dim])
        emb = emb * mask_emb
        emb = fluid.layers.reshape(
            emb, shape=[-1, 1, self.seq_len, self.hid_dim])
        conv_3 = self._simple_conv_pool_1(emb)
        fc_1 = self._fc1(conv_3)
        prediction = self._fc_prediction(fc_1)

        if label:
            cost = fluid.layers.cross_entropy(input=prediction, label=label)
            avg_cost = fluid.layers.mean(x=cost)
            acc = fluid.layers.accuracy(input=prediction, label=label)

            return avg_cost, prediction, acc
        else:
            return prediction
