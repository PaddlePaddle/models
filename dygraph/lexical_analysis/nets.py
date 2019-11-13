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
from paddle.fluid.dygraph import GRUUnit
from paddle.fluid.dygraph.base import to_variable
import numpy as np


class DynamicGRU(fluid.dygraph.Layer):
    def __init__(self,
                 scope_name,
                 size,
                 param_attr=None,
                 bias_attr=None,
                 is_reverse=False,
                 gate_activation='sigmoid',
                 candidate_activation='tanh',
                 h_0=None,
                 origin_mode=False,
                 init_size=None):
        super(DynamicGRU, self).__init__(scope_name)
        self.gru_unit = GRUUnit(
            self.full_name(),
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
            input_ = inputs[:, i:i + 1, :]
            input_ = fluid.layers.reshape(
                input_, [-1, input_.shape[2]], inplace=False)
            hidden, reset, gate = self.gru_unit(input_, hidden)
            hidden_ = fluid.layers.reshape(
                hidden, [-1, 1, hidden.shape[1]], inplace=False)
            res.append(hidden_)
        if self.is_reverse:
            res = res[::-1]
        res = fluid.layers.concat(res, axis=1)
        return res


class LAC(fluid.dygraph.Layer):
    def __init__(self,
                 name_scope,
                 args,
                 vocab_size,
                 num_labels,
                 for_infer=True,
                 target=None):
        super(LAC, self).__init__(name_scope)
        self.word_emb_dim = args.word_emb_dim
        self.dict_dim = vocab_size
        self.grnn_hidden_dim = args.grnn_hidden_dim
        self.emb_lr = args.emb_learning_rate if 'emb_learning_rate' in dir(
            args) else 1.0
        self.crf_lr = args.emb_learning_rate if 'crf_learning_rate' in dir(
            args) else 1.0
        self.bigru_num = args.bigru_num
        self.init_bound = 0.1
        self.IS_SPARSE = True
        self.max_seq_lens = args.max_seq_lens
        self.grnn_hidden_dim = args.grnn_hidden_dim
        self._word_embedding = Embedding(
            self.full_name(),
            size=[vocab_size, self.word_emb_dim],
            dtype='float32',
            is_sparse=self.IS_SPARSE,
            param_attr=fluid.ParamAttr(
                learning_rate=self.emb_lr,
                initializer=fluid.initializer.Uniform(
                    low=-self.init_bound, high=self.init_bound)))
        self._emission_fc = FC(
            self.full_name(),
            size=num_labels,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Uniform(
                    low=-self.init_bound, high=self.init_bound),
                regularizer=fluid.regularizer.L2DecayRegularizer(
                    regularization_coeff=1e-4)))

    def _bigru_layer(input_feature, grnn_hidden_dim):
        """
        define the bidirectional gru layer
        """
        pre_gru = FC(input=input_feature,
                     size=grnn_hidden_dim * 3,
                     param_attr=fluid.ParamAttr(
                         initializer=fluid.initializer.Uniform(
                             low=-init_bound, high=init_bound),
                         regularizer=fluid.regularizer.L2DecayRegularizer(
                             regularization_coeff=1e-4)))
        gru = DynamicGRU(
            input=pre_gru,
            size=grnn_hidden_dim,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Uniform(
                    low=-init_bound, high=init_bound),
                regularizer=fluid.regularizer.L2DecayRegularizer(
                    regularization_coeff=1e-4)))
        pre_gru_r = FC(input=input_feature,
                       size=grnn_hidden_dim * 3,
                       param_attr=fluid.ParamAttr(
                           initializer=fluid.initializer.Uniform(
                               low=-init_bound, high=init_bound),
                           regularizer=fluid.regularizer.L2DecayRegularizer(
                               regularization_coeff=1e-4)))
        gru_r = DynamicGRU(
            input=pre_gru_r,
            size=grnn_hidden_dim,
            is_reverse=True,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Uniform(
                    low=-init_bound, high=init_bound),
                regularizer=fluid.regularizer.L2DecayRegularizer(
                    regularization_coeff=1e-4)))
        bi_merge = fluid.layers.concat(input=[gru, gru_r], axis=1)
        return bi_merge

    def forward(self, inputs, targets, seq_lens):
        emb = self._word_embedding(inputs)
        o_np_mask = (inputs.numpy() != self.dict_dim).astype('float32')
        mask_emb = fluid.layers.expand(
            to_variable(o_np_mask), [1, self.word_emb_dim])
        emb = emb * mask_emb
        emb = fluid.layers.reshape(
            emb, shape=[-1, 1, self.max_seq_lens, self.hid_dim])
        input_feature = emb
        for i in range(self.bigru_num):
            bigru_output = _bigru_layer(input_feature, self._grnn_hidden_dim)
            input_feature = bigru_output
        emission = self_emission_fc(input_feature)

        if targets is not None:
            crf_cost = fluid.layers.linear_chain_crf(
                input=emission,
                label=target,
                param_attr=fluid.ParamAttr(
                    name='crfw', learning_rate=crf_lr),
                length=seq_lens)
            avg_cost = fluid.layers.mean(x=crf_cost)
            crf_decode = fluid.layers.crf_decoding(
                input=emission,
                param_attr=fluid.ParamAttr(name='crfw'),
                length=seq_lens)
            return avg_cost, crf_decode

        else:
            size = emission.shape[1]
            fluid.layers.create_parameter(
                shape=[size + 2, size], dtype=emission.dtype, name='crfw')
            crf_decode = fluid.layers.crf_decoding(
                input=emission,
                param_attr=fluid.ParamAttr(name='crfw'),
                length=seq_lens)
            return crf_decode
