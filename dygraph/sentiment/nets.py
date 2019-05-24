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
                 num_channels,
                 num_filters,
                 filter_size,
                 pool_size=-1,
                 pool_stride=1,
                 pool_padding=0,
                 pool_type='max',
                 global_pooling=False,
                 use_cudnn=False,
                 batch_size=None):
        super(SimpleConvPool, self).__init__(name_scope)
        self.batch_size = batch_size
        self._conv2d = Conv2D(
            self.full_name(),
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=filter_size,
            padding=[0, 1],
            use_cudnn=use_cudnn)

        self._pool2d = Pool2D(
            self.full_name(),
            pool_size=pool_size,
            pool_type=pool_type,
            pool_stride=pool_stride,
            pool_padding=pool_padding,
            global_pooling=global_pooling,
            use_cudnn=use_cudnn)

    def forward(self, inputs, mask):
        x = self._conv2d(inputs)
        x = x * mask
        x = fluid.layers.reduce_max(x, dim=-1)
        x = fluid.layers.reshape(x, shape=[self.batch_size, -1])
        return x


class CNN(fluid.dygraph.Layer):
    def __init__(self, name_scope, dict_dim, batch_size, seq_len):
        super(CNN, self).__init__(name_scope)
        self.dict_dim = dict_dim
        self.emb_dim = 128
        self.hid_dim = 128
        self.hid_dim2 = 96
        self.class_dim = 2
        self.win_size = [1, 3]
        self.batch_size = batch_size
        self.seq_len = seq_len
        init_scale = 0.1
        self.embedding = Embedding(
            self.full_name(),
            size=[self.dict_dim, self.emb_dim],
            dtype='float32',
            is_sparse=False,
            param_attr=fluid.ParamAttr(
                name='embedding_para',
                initializer=fluid.initializer.UniformInitializer(
                    low=-init_scale, high=init_scale)))

        self._simple_conv_pool_1 = SimpleConvPool(
            self.full_name(),
            self.emb_dim,
            self.hid_dim,
            self.win_size,
            2,
            2,
            batch_size=batch_size)
        self._fc1 = FC(self.full_name(), size=self.hid_dim2, act="softmax")
        self._fc_prediction = FC(self.full_name(),
                                 size=self.class_dim,
                                 act="softmax")

    def forward(self, inputs, label=None):
        np_mask = (inputs.numpy() > 0).reshape((self.batch_size, 1, 1,
                                                -1)).astype('float32')
        mask = to_variable(np_mask)
        mask = fluid.layers.expand(mask, [1, self.hid_dim, 1, 1])
        emb = self.embedding(inputs)
        emb = fluid.layers.reshape(
            emb, shape=[-1, 1, self.seq_len, self.hid_dim])
        emb = fluid.layers.transpose(emb, [0, 3, 1, 2])
        conv_3 = self._simple_conv_pool_1(emb, mask)

        fc_1 = self._fc1(conv_3)
        prediction = self._fc_prediction(fc_1)

        if label:
            cost = fluid.layers.cross_entropy(input=prediction, label=label)
            avg_cost = fluid.layers.mean(x=cost)
            acc = fluid.layers.accuracy(input=prediction, label=label)

            return avg_cost, prediction, acc
        else:
            return prediction