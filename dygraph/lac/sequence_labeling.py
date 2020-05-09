#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
"""
The function lex_net(args) define the lexical analysis network structure
"""
import sys
import os
import math
import numpy as np

import paddle.fluid as fluid
from paddle.fluid.initializer import NormalInitializer
from paddle.fluid.dygraph import to_variable
from paddle.fluid.dygraph.nn import Embedding, Linear, GRUUnit

class DynamicGRU(fluid.dygraph.Layer):
    def __init__(self,
                 size,
                 h_0=None,
                 param_attr=None,
                 bias_attr=None,
                 is_reverse=False,
                 gate_activation='sigmoid',
                 candidate_activation='tanh',
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


class BiGRU(fluid.dygraph.Layer):
    def __init__(self,
                 input_dim,
                 grnn_hidden_dim,
                 init_bound,
                 h_0=None):
        super(BiGRU, self).__init__()

        self.pre_gru = Linear(input_dim=input_dim,
                            output_dim=grnn_hidden_dim * 3,
                            param_attr=fluid.ParamAttr(
                            initializer=fluid.initializer.Uniform(
                                low=-init_bound, high=init_bound),
                                regularizer=fluid.regularizer.L2DecayRegularizer(
                                    regularization_coeff=1e-4)))#,
                            #num_flatten_dims=2)

        self.gru = DynamicGRU(size=grnn_hidden_dim,
                h_0=h_0,
                param_attr=fluid.ParamAttr(
                    initializer=fluid.initializer.Uniform(
                        low=-init_bound, high=init_bound),
                    regularizer=fluid.regularizer.L2DecayRegularizer(
                        regularization_coeff=1e-4)))

        self.pre_gru_r = Linear(input_dim=input_dim,
                            output_dim=grnn_hidden_dim * 3,
                            param_attr=fluid.ParamAttr(
                                initializer=fluid.initializer.Uniform(
                                    low=-init_bound, high=init_bound),
                                regularizer=fluid.regularizer.L2DecayRegularizer(
                                    regularization_coeff=1e-4)))#,
                            #num_flatten_dims=2)

        self.gru_r = DynamicGRU(size=grnn_hidden_dim,
                            is_reverse=True,
                            h_0=h_0,
                            param_attr=fluid.ParamAttr(
                                initializer=fluid.initializer.Uniform(
                                    low=-init_bound, high=init_bound),
                                regularizer=fluid.regularizer.L2DecayRegularizer(
                                    regularization_coeff=1e-4)))


    def forward(self, input_feature):
        res_pre_gru = self.pre_gru(input_feature)
        res_gru = self.gru(res_pre_gru)
        res_pre_gru_r = self.pre_gru_r(input_feature)
        res_gru_r = self.gru_r(res_pre_gru_r)
        bi_merge = fluid.layers.concat(input=[res_gru, res_gru_r], axis=-1)
        return bi_merge


class Linear_chain_crf(fluid.dygraph.Layer):

    def __init__(self,
                param_attr, 
                size=None,
                is_test=False,
                dtype='float32'):
        super(Linear_chain_crf, self).__init__()

        self._param_attr = param_attr
        self._dtype = dtype
        self._size = size
        self._is_test=is_test
        self._transition = self.create_parameter(
                        attr=self._param_attr,
                        shape=[self._size + 2, self._size],
                        dtype=self._dtype)

    @property
    def weight(self):
        return self._transition

    @weight.setter
    def weight(self, value):
        self._transition = value

    def forward(self, input, label, length=None):
        
        alpha = self._helper.create_variable_for_type_inference(
                        dtype=self._dtype)
        emission_exps = self._helper.create_variable_for_type_inference(
                        dtype=self._dtype)
        transition_exps = self._helper.create_variable_for_type_inference(
                        dtype=self._dtype)
        log_likelihood = self._helper.create_variable_for_type_inference(
                        dtype=self._dtype)
        this_inputs = {
            "Emission": [input],
            "Transition": self._transition,
            "Label": [label]
        }
        if length is not None:
            this_inputs['Length'] = [length]
        self._helper.append_op(
                        type='linear_chain_crf',
                        inputs=this_inputs,
                        outputs={
                            "Alpha": [alpha],
                            "EmissionExps": [emission_exps],
                            "TransitionExps": transition_exps,
                            "LogLikelihood": log_likelihood
                        },
                        attrs={
                            "is_test": self._is_test,
                        })
        return log_likelihood


class Crf_decoding(fluid.dygraph.Layer):

    def __init__(self,
                param_attr, 
                size=None,
                is_test=False,
                dtype='float32'):
        super(Crf_decoding, self).__init__()

        self._dtype = dtype
        self._size = size
        self._is_test = is_test
        self._param_attr = param_attr
        self._transition = self.create_parameter(
                        attr=self._param_attr,
                        shape=[self._size + 2, self._size],
                        dtype=self._dtype)

    @property
    def weight(self):
        return self._transition

    @weight.setter
    def weight(self, value):
        self._transition = value

    def forward(self, input, label=None, length=None):
        
        viterbi_path = self._helper.create_variable_for_type_inference(
                        dtype=self._dtype)
        this_inputs = {"Emission": [input], "Transition": self._transition, "Label": label}
        if length is not None:
            this_inputs['Length'] = [length]
        self._helper.append_op(
                        type='crf_decoding',
                        inputs=this_inputs,
                        outputs={"ViterbiPath": [viterbi_path]},
                        attrs={
                            "is_test": self._is_test,
                        })
        return viterbi_path


class Chunk_eval(fluid.dygraph.Layer):

    def __init__(self,
                num_chunk_types,
                chunk_scheme,
                excluded_chunk_types=None):
        super(Chunk_eval, self).__init__()
        self.num_chunk_types = num_chunk_types
        self.chunk_scheme = chunk_scheme
        self.excluded_chunk_types = excluded_chunk_types

    def forward(self, input, label, seq_length=None):
        
        precision = self._helper.create_variable_for_type_inference(dtype="float32")
        recall = self._helper.create_variable_for_type_inference(dtype="float32")
        f1_score = self._helper.create_variable_for_type_inference(dtype="float32")
        num_infer_chunks = self._helper.create_variable_for_type_inference(dtype="int64")
        num_label_chunks = self._helper.create_variable_for_type_inference(dtype="int64")
        num_correct_chunks = self._helper.create_variable_for_type_inference(dtype="int64")

        this_input = {"Inference": [input], "Label": [label]}
        if seq_length is not None:
            this_input["SeqLength"] = [seq_length]

        self._helper.append_op(
                        type='chunk_eval',
                        inputs=this_input,
                        outputs={
                                "Precision": [precision],
                                "Recall": [recall],
                                "F1-Score": [f1_score],
                                "NumInferChunks": [num_infer_chunks],
                                "NumLabelChunks": [num_label_chunks],
                                "NumCorrectChunks": [num_correct_chunks]
                            },
                        attrs={
                            "num_chunk_types": self.num_chunk_types,
                            "chunk_scheme": self.chunk_scheme,
                            "excluded_chunk_types": self.excluded_chunk_types or []
                        })
        return (precision, recall, f1_score, num_infer_chunks, num_label_chunks,
            num_correct_chunks)


class lex_net(fluid.dygraph.Layer):
    def __init__(self, 
                    args, 
                    vocab_size, 
                    num_labels,
                    length=None):
        super(lex_net, self).__init__()
        """
        define the lexical analysis network structure
        word: stores the input of the model
        for_infer: a boolean value, indicating if the model to be created is for training or predicting.

        return:
            for infer: return the prediction
            otherwise: return the prediction
        """
        self.word_emb_dim = args.word_emb_dim
        self.vocab_size = vocab_size
        self.num_labels = num_labels
        self.grnn_hidden_dim = args.grnn_hidden_dim
        self.emb_lr = args.emb_learning_rate if 'emb_learning_rate' in dir(args) else 1.0
        self.crf_lr = args.emb_learning_rate if 'crf_learning_rate' in dir(args) else 1.0
        self.bigru_num = args.bigru_num
        self.init_bound = 0.1
        #self.IS_SPARSE = True

        self.word_embedding = Embedding(
            size=[self.vocab_size, self.word_emb_dim],
            dtype='float32',
            #is_sparse=self.IS_SPARSE,
            param_attr=fluid.ParamAttr(
                learning_rate=self.emb_lr,
                name="word_emb",
                initializer=fluid.initializer.Uniform(
                    low=-self.init_bound, high=self.init_bound)))

        h_0 = np.zeros((args.batch_size, self.grnn_hidden_dim), dtype="float32")
        h_0 = to_variable(h_0)
        self.bigru_units = []
        for i in range(self.bigru_num):
            if i == 0:
                self.bigru_units.append(
                    self.add_sublayer("bigru_units%d" % i,
                    BiGRU(self.grnn_hidden_dim, self.grnn_hidden_dim, self.init_bound, h_0=h_0)
                ))
            else:
                self.bigru_units.append(
                    self.add_sublayer("bigru_units%d" % i,
                    BiGRU(self.grnn_hidden_dim * 2, self.grnn_hidden_dim, self.init_bound, h_0=h_0)
                ))
        
        self.fc = Linear(input_dim=self.grnn_hidden_dim * 2,
                        output_dim=self.num_labels,
                        param_attr=fluid.ParamAttr(
                            initializer=fluid.initializer.Uniform(
                                low=-self.init_bound, high=self.init_bound),
                            regularizer=fluid.regularizer.L2DecayRegularizer(
                                regularization_coeff=1e-4)))#,
                        #num_flatten_dims=2)
        
        self.linear_chain_crf = Linear_chain_crf(
                param_attr=fluid.ParamAttr(
                    name='linear_chain_crfw', learning_rate=self.crf_lr),
                size=self.num_labels)

        self.crf_decoding = Crf_decoding(
                param_attr=fluid.ParamAttr(
                    name='crfw', learning_rate=self.crf_lr),
                size=self.num_labels)
        
    def forward(self, word, target=None, length=None):
        """
        Configure the network
        """
        #word = fluid.layers.unsqueeze(word, [2])
        word_embed = self.word_embedding(word)
        input_feature = word_embed
        
        for i in range(self.bigru_num):
            bigru_output = self.bigru_units[i](input_feature)
            input_feature = bigru_output

        emission = self.fc(bigru_output)

        if target is not None:
            crf_cost = self.linear_chain_crf(
                input=emission,
                label=target,
                length=length)
            avg_cost = fluid.layers.mean(x=crf_cost)
            self.crf_decoding.weight = self.linear_chain_crf.weight
            crf_decode = self.crf_decoding(
                input=emission,
                length=length)
            return avg_cost, crf_decode#, word_embed, bigru_output, emission
        else:
            crf_decode = self.crf_decoding(
                input=emission,
                length=length)
            return crf_decode


    
