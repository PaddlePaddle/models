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
import numpy as np


class SimpleLSTMRNN(fluid.Layer):
    def __init__(self,
                 name_scope,
                 hidden_size,
                 num_steps,
                 num_layers=2,
                 init_scale=0.1,
                 dropout=None):
        super(SimpleLSTMRNN, self).__init__(name_scope)
        self._hidden_size = hidden_size
        self._num_layers = num_layers
        self._init_scale = init_scale
        self._dropout = dropout
        self._input = None
        self._num_steps = num_steps
        self.cell_array = []
        self.hidden_array = []

        self.weight_1_arr = []
        self.weight_2_arr = []
        self.bias_arr = []
        self.mask_array = []

        for i in range(self._num_layers):
            weight_1 = self.create_parameter(
                attr=fluid.ParamAttr(
                    initializer=fluid.initializer.UniformInitializer(
                        low=-self._init_scale, high=self._init_scale)),
                shape=[self._hidden_size * 2, self._hidden_size * 4],
                dtype="float32",
                default_initializer=fluid.initializer.UniformInitializer(
                    low=-self._init_scale, high=self._init_scale))
            self.weight_1_arr.append(self.add_parameter('w_%d' % i, weight_1))
            bias_1 = self.create_parameter(
                attr=fluid.ParamAttr(
                    initializer=fluid.initializer.UniformInitializer(
                        low=-self._init_scale, high=self._init_scale)),
                shape=[self._hidden_size * 4],
                dtype="float32",
                default_initializer=fluid.initializer.Constant(0.0))
            self.bias_arr.append(self.add_parameter('b_%d' % i, bias_1))

    def forward(self, input_embedding, init_hidden=None, init_cell=None):
        self.cell_array = []
        self.hidden_array = []

        for i in range(self._num_layers):
            pre_hidden = fluid.layers.slice(
                init_hidden, axes=[0], starts=[i], ends=[i + 1])
            pre_cell = fluid.layers.slice(
                init_cell, axes=[0], starts=[i], ends=[i + 1])
            pre_hidden = fluid.layers.reshape(
                pre_hidden, shape=[-1, self._hidden_size])
            pre_cell = fluid.layers.reshape(
                pre_cell, shape=[-1, self._hidden_size])
            self.hidden_array.append(pre_hidden)
            self.cell_array.append(pre_cell)

        res = []
        for index in range(self._num_steps):
            self._input = fluid.layers.slice(
                input_embedding, axes=[1], starts=[index], ends=[index + 1])
            self._input = fluid.layers.reshape(
                self._input, shape=[-1, self._hidden_size])
            for k in range(self._num_layers):
                pre_hidden = self.hidden_array[k]
                pre_cell = self.cell_array[k]
                weight_1 = self.weight_1_arr[k]
                bias = self.bias_arr[k]

                nn = fluid.layers.concat([self._input, pre_hidden], 1)
                gate_input = fluid.layers.matmul(x=nn, y=weight_1)

                gate_input = fluid.layers.elementwise_add(gate_input, bias)
                i, j, f, o = fluid.layers.split(
                    gate_input, num_or_sections=4, dim=-1)
                c = pre_cell * fluid.layers.sigmoid(f) + fluid.layers.sigmoid(
                    i) * fluid.layers.tanh(j)
                m = fluid.layers.tanh(c) * fluid.layers.sigmoid(o)
                self.hidden_array[k] = m
                self.cell_array[k] = c
                self._input = m

                if self._dropout is not None and self._dropout > 0.0:
                    self._input = fluid.layers.dropout(
                        self._input,
                        dropout_prob=self._dropout,
                        dropout_implementation='upscale_in_train')
            res.append(
                fluid.layers.reshape(
                    self._input, shape=[1, -1, self._hidden_size]))
        real_res = fluid.layers.concat(res, 0)
        real_res = fluid.layers.transpose(x=real_res, perm=[1, 0, 2])
        last_hidden = fluid.layers.concat(self.hidden_array, 1)
        last_hidden = fluid.layers.reshape(
            last_hidden, shape=[-1, self._num_layers, self._hidden_size])
        last_hidden = fluid.layers.transpose(x=last_hidden, perm=[1, 0, 2])
        last_cell = fluid.layers.concat(self.cell_array, 1)
        last_cell = fluid.layers.reshape(
            last_cell, shape=[-1, self._num_layers, self._hidden_size])
        last_cell = fluid.layers.transpose(x=last_cell, perm=[1, 0, 2])
        return real_res, last_hidden, last_cell


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


class BOW(fluid.dygraph.Layer):
    def __init__(self, name_scope, dict_dim, batch_size, seq_len):
        super(BOW, self).__init__(name_scope)
        self.dict_dim = dict_dim
        self.emb_dim = 128
        self.hid_dim = 128
        self.fc_hid_dim = 96
        self.class_dim = 2
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.embedding = Embedding(
            self.full_name(),
            size=[self.dict_dim + 1, self.emb_dim],
            dtype='float32',
            is_sparse=False)
        self._fc1 = FC(self.full_name(), size=self.fc_hid_dim, act="tanh")
        self._fc2 = FC(self.full_name(), size=self.class_dim, act="tanh")
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
        bow_1 = fluid.layers.reduce_sum(emb, dim=1)
        bow_1 = fluid.layers.tanh(bow_1)
        fc_1 = self._fc1(bow_1)
        fc_2 = self._fc2(fc_1)
        prediction = self._fc_prediction(fc_2)
        if label:
            cost = fluid.layers.cross_entropy(input=prediction, label=label)
            avg_cost = fluid.layers.mean(x=cost)
            acc = fluid.layers.accuracy(input=prediction, label=label)

            return avg_cost, prediction, acc
        else:
            return prediction


class LSTM(fluid.dygraph.Layer):
    def __init__(self, name_scope, dict_dim, batch_size, seq_len):
        super(LSTM, self).__init__(name_scope)
        self.dict_dim = dict_dim
        self.emb_dim = 128
        self.hid_dim = 128
        self.fc_hid_dim = 96
        self.class_dim = 2
        self.lstm_num_steps = 60
        self.lstm_num_layers = 1
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.embedding = Embedding(
            self.full_name(),
            size=[self.dict_dim + 1, self.emb_dim],
            dtype='float32',
            param_attr=fluid.ParamAttr(learning_rate=30),
            is_sparse=False)
        self._fc1 = FC(self.full_name(), size=self.hid_dim, num_flatten_dims=2)
        self._fc2 = FC(self.full_name(), size=self.fc_hid_dim, act="tanh")
        self._fc_prediction = FC(self.full_name(),
                                 size=self.class_dim,
                                 act="softmax")
        self.simple_lstm_rnn = SimpleLSTMRNN(
            self.full_name(),
            self.hid_dim,
            num_steps=self.lstm_num_steps,
            num_layers=self.lstm_num_layers,
            init_scale=0.1,
            dropout=None)

    def forward(self, inputs, init_hidden, init_cell, label=None):
        emb = self.embedding(inputs)
        o_np_mask = (inputs.numpy() != self.dict_dim).astype('float32')
        mask_emb = fluid.layers.expand(
            to_variable(o_np_mask), [1, self.hid_dim])
        emb = emb * mask_emb
        emb = fluid.layers.reshape(emb, shape=[-1, self.seq_len, self.hid_dim])
        fc_1 = self._fc1(emb)
        init_h = fluid.layers.reshape(
            init_hidden, shape=[self.lstm_num_layers, -1, self.hid_dim])
        init_c = fluid.layers.reshape(
            init_cell, shape=[self.lstm_num_layers, -1, self.hid_dim])
        real_res, last_hidden, last_cell = self.simple_lstm_rnn(fc_1, init_h,
                                                                init_c)
        last_hidden = fluid.layers.reshape(
            last_hidden, shape=[-1, self.hid_dim])
        tanh_1 = fluid.layers.tanh(last_hidden)
        fc_2 = self._fc2(tanh_1)
        prediction = self._fc_prediction(fc_2)
        if label:
            cost = fluid.layers.cross_entropy(input=prediction, label=label)
            avg_cost = fluid.layers.mean(x=cost)
            acc = fluid.layers.accuracy(input=prediction, label=label)

            return avg_cost, prediction, acc, last_hidden, last_cell
        else:
            return prediction
