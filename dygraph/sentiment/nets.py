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
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, Linear, Embedding
from paddle.fluid.dygraph import GRUUnit
from paddle.fluid.dygraph.base import to_variable
import numpy as np

    
class DynamicGRU(fluid.dygraph.Layer):
    def __init__(self,
                 size,
                 param_attr=None,
                 bias_attr=None,
                 is_reverse=False,
                 gate_activation='sigmoid',
                 candidate_activation='tanh',
                 h_0=None,
                 origin_mode=False,
                 init_size = None):
        super(DynamicGRU, self).__init__()
        self.gru_unit = GRUUnit(
            size * 3,
            param_attr=param_attr,
            bias_attr=bias_attr,
            activation=candidate_activation,
            gate_activation=gate_activation,
            origin_mode=origin_mode)
        self.size = size
        self.h_0 = h_0
        self.is_reverse = is_reverse
    def forward(self, inputs):
        hidden = self.h_0
        res = []
        for i in range(inputs.shape[1]):
            if self.is_reverse:
                i = inputs.shape[1] - 1 - i
            input_ = inputs[ :, i:i+1, :]
            input_ = fluid.layers.reshape(input_, [-1, input_.shape[2]], inplace=False)
            hidden, reset, gate = self.gru_unit(input_, hidden)
            hidden_ = fluid.layers.reshape(hidden, [-1, 1, hidden.shape[1]], inplace=False)
            res.append(hidden_)
        if self.is_reverse:
            res = res[::-1]
        res = fluid.layers.concat(res, axis=1)
        return res

class SimpleConvPool(fluid.dygraph.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 filter_size,
                 use_cudnn=False,
                 batch_size=None):
        super(SimpleConvPool, self).__init__()
        self.batch_size = batch_size
        self._conv2d = Conv2D(num_channels = num_channels,
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
    def __init__(self,  dict_dim, batch_size, seq_len):
        super(CNN, self).__init__()
        self.dict_dim = dict_dim
        self.emb_dim = 128
        self.hid_dim = 128
        self.fc_hid_dim = 96
        self.class_dim = 2
        self.channels = 1
        self.win_size = [3, self.hid_dim]
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.embedding = Embedding(
            size=[self.dict_dim + 1, self.emb_dim],
            dtype='float32',
            is_sparse=False)
        self._simple_conv_pool_1 = SimpleConvPool(
            self.channels,
            self.hid_dim,
            self.win_size,
            batch_size=self.batch_size)
        self._fc1 = Linear(input_dim = self.hid_dim*self.seq_len, output_dim=self.fc_hid_dim, act="softmax")
        self._fc_prediction = Linear(input_dim = self.fc_hid_dim,
                                 output_dim = self.class_dim,
                                 act="softmax")
    def forward(self, inputs, label=None):
        emb = self.embedding(inputs)
        o_np_mask = (inputs.numpy().reshape(-1,1) != self.dict_dim).astype('float32')
        mask_emb = fluid.layers.expand(
            to_variable(o_np_mask), [1, self.hid_dim])
        emb = emb * mask_emb
        emb = fluid.layers.reshape(
            emb, shape=[-1, self.channels , self.seq_len, self.hid_dim])
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
    def __init__(self, dict_dim, batch_size, seq_len):
        super(BOW, self).__init__()
        self.dict_dim = dict_dim
        self.emb_dim = 128
        self.hid_dim = 128
        self.fc_hid_dim = 96
        self.class_dim = 2
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.embedding = Embedding(
            size=[self.dict_dim + 1, self.emb_dim],
            dtype='float32',
            is_sparse=False)
        self._fc1 = Linear(input_dim = self.hid_dim, output_dim=self.hid_dim, act="tanh")
        self._fc2 = Linear(input_dim = self.hid_dim, output_dim=self.fc_hid_dim, act="tanh")
        self._fc_prediction = Linear(input_dim = self.fc_hid_dim,
                                 output_dim = self.class_dim,
                                 act="softmax")
    def forward(self, inputs, label=None):
        emb = self.embedding(inputs)
        o_np_mask = (inputs.numpy().reshape(-1,1) != self.dict_dim).astype('float32')
        mask_emb = fluid.layers.expand(
            to_variable(o_np_mask), [1, self.hid_dim])
        emb = emb * mask_emb
        emb = fluid.layers.reshape(
            emb, shape=[-1, self.seq_len, self.hid_dim])
        bow_1 = fluid.layers.reduce_sum(emb, dim=1)
        bow_1 = fluid.layers.tanh(bow_1)
        fc_1 = self._fc1(bow_1)
        fc_2 = self._fc2(fc_1)
        prediction = self._fc_prediction(fc_2)
        if label is not None:
            cost = fluid.layers.cross_entropy(input=prediction, label=label)
            avg_cost = fluid.layers.mean(x=cost)
            acc = fluid.layers.accuracy(input=prediction, label=label)
            return avg_cost, prediction, acc
        else:
            return prediction


class GRU(fluid.dygraph.Layer):
    def __init__(self, dict_dim, batch_size, seq_len):
        super(GRU, self).__init__()
        self.dict_dim = dict_dim
        self.emb_dim = 128
        self.hid_dim = 128
        self.fc_hid_dim = 96
        self.class_dim = 2
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.embedding = Embedding(
            size=[self.dict_dim + 1, self.emb_dim],
            dtype='float32',
            param_attr=fluid.ParamAttr(learning_rate=30),
            is_sparse=False)
        h_0 = np.zeros((self.batch_size, self.hid_dim), dtype="float32")
        h_0 = to_variable(h_0)
        self._fc1 = Linear(input_dim = self.hid_dim, output_dim=self.hid_dim*3)
        self._fc2 = Linear(input_dim=self.hid_dim, output_dim=self.fc_hid_dim, act="tanh")
        self._fc_prediction = Linear(input_dim=self.fc_hid_dim,
                                 output_dim=self.class_dim,
                                 act="softmax")
        self._gru = DynamicGRU( size= self.hid_dim, h_0=h_0)
    def forward(self, inputs, label=None):
        emb = self.embedding(inputs)
        o_np_mask =to_variable(inputs.numpy().reshape(-1,1) != self.dict_dim).astype('float32')
        mask_emb = fluid.layers.expand(
            to_variable(o_np_mask), [1, self.hid_dim])
        emb = emb * mask_emb
        emb = fluid.layers.reshape(emb, shape=[self.batch_size, -1, self.hid_dim])
        fc_1 = self._fc1(emb)
        gru_hidden = self._gru(fc_1)
        gru_hidden = fluid.layers.reduce_max(gru_hidden, dim=1)
        tanh_1 = fluid.layers.tanh(gru_hidden)
        fc_2 = self._fc2(tanh_1)
        prediction = self._fc_prediction(fc_2)
        if label:
            cost = fluid.layers.cross_entropy(input=prediction, label=label)
            avg_cost = fluid.layers.mean(x=cost)
            acc = fluid.layers.accuracy(input=prediction, label=label)
            return avg_cost, prediction, acc
        else:
            return prediction

        
class BiGRU(fluid.dygraph.Layer):
    def __init__(self, dict_dim, batch_size, seq_len):
        super(BiGRU, self).__init__()
        self.dict_dim = dict_dim
        self.emb_dim = 128
        self.hid_dim = 128
        self.fc_hid_dim = 96
        self.class_dim = 2
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.embedding = Embedding(
            size=[self.dict_dim + 1, self.emb_dim],
            dtype='float32',
            param_attr=fluid.ParamAttr(learning_rate=30),
            is_sparse=False)
        h_0 = np.zeros((self.batch_size, self.hid_dim), dtype="float32")
        h_0 = to_variable(h_0)
        self._fc1 = Linear(input_dim = self.hid_dim, output_dim=self.hid_dim*3)
        self._fc2 = Linear(input_dim = self.hid_dim*2, output_dim=self.fc_hid_dim, act="tanh")
        self._fc_prediction = Linear(input_dim=self.fc_hid_dim,
                                 output_dim=self.class_dim,
                                 act="softmax")
        self._gru_forward = DynamicGRU( size= self.hid_dim, h_0=h_0,is_reverse=False)
        self._gru_backward = DynamicGRU( size= self.hid_dim, h_0=h_0,is_reverse=True)

    def forward(self, inputs, label=None):
        emb = self.embedding(inputs)
        o_np_mask =to_variable(inputs.numpy() .reshape(-1,1)!= self.dict_dim).astype('float32')
        mask_emb = fluid.layers.expand(
            to_variable(o_np_mask), [1, self.hid_dim])
        emb = emb * mask_emb
        emb = fluid.layers.reshape(emb, shape=[self.batch_size, -1, self.hid_dim])
        fc_1 = self._fc1(emb)
        gru_forward = self._gru_forward(fc_1)
        gru_backward = self._gru_backward(fc_1)
        gru_forward_tanh = fluid.layers.tanh(gru_forward)
        gru_backward_tanh = fluid.layers.tanh(gru_backward)
        encoded_vector = fluid.layers.concat(
            input=[gru_forward_tanh, gru_backward_tanh], axis=2)
        encoded_vector = fluid.layers.reduce_max(encoded_vector, dim=1)
        fc_2 = self._fc2(encoded_vector)
        prediction = self._fc_prediction(fc_2)
        if label:
            cost = fluid.layers.cross_entropy(input=prediction, label=label)
            avg_cost = fluid.layers.mean(x=cost)
            acc = fluid.layers.accuracy(input=prediction, label=label)
            return avg_cost, prediction, acc
        else:
            return prediction
