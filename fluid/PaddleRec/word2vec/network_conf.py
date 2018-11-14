# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
neural network for word2vec
"""

from __future__ import print_function

import math
import numpy as np

import paddle.fluid as fluid

def skip_gram_word2vec(dict_size, word_frequencys, embedding_size):
    def nce_layer(input, label, embedding_size, num_total_classes, num_neg_samples, sampler, custom_dist, sample_weight):
        # convert word_frequencys to tensor
        nid_freq_arr = np.array(word_frequencys).astype('float32')
        nid_freq_var = fluid.layers.assign(input=nid_freq_arr)

        w_param_name = "nce_w"
        b_param_name = "nce_b"
        w_param = fluid.default_main_program().global_block().create_parameter(
            shape=[num_total_classes, embedding_size], dtype='float32', name=w_param_name)
        b_param = fluid.default_main_program().global_block().create_parameter(
            shape=[num_total_classes, 1], dtype='float32', name=b_param_name)

        cost = fluid.layers.nce(
            input=input,
            label=label,
            num_total_classes=num_total_classes,
            sampler=sampler,
            custom_dist=nid_freq_var,
            sample_weight = sample_weight,
            param_attr=fluid.ParamAttr(name=w_param_name),
            bias_attr=fluid.ParamAttr(name=b_param_name),
            num_neg_samples=num_neg_samples)

        return cost

    input_word = fluid.layers.data(name="input_word", shape=[1], dtype='int64')
    predict_word = fluid.layers.data(name='predict_word', shape=[1], dtype='int64')
    data_list = [input_word, predict_word]

    emb = fluid.layers.embedding(
        input=input_word,
        size=[dict_size, embedding_size],
        param_attr=fluid.ParamAttr(initializer=fluid.initializer.Normal(scale=1 / math.sqrt(dict_size))))

    cost = nce_layer(emb, predict_word, embedding_size, dict_size, 5, "custom_dist", word_frequencys, None)
    avg_cost = fluid.layers.reduce_mean(cost)

    return avg_cost, data_list
