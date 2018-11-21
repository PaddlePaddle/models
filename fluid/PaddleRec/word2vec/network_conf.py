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


def skip_gram_word2vec(dict_size,
                       word_frequencys,
                       embedding_size,
                       max_code_length=None,
                       with_hsigmoid=False,
                       with_nce=True,
                       is_sparse=False):
    def nce_layer(input, label, embedding_size, num_total_classes,
                  num_neg_samples, sampler, word_frequencys, sample_weight):

        w_param_name = "nce_w"
        b_param_name = "nce_b"
        w_param = fluid.default_main_program().global_block().create_parameter(
            shape=[num_total_classes, embedding_size],
            dtype='float32',
            name=w_param_name)
        b_param = fluid.default_main_program().global_block().create_parameter(
            shape=[num_total_classes, 1], dtype='float32', name=b_param_name)

        cost = fluid.layers.nce(input=input,
                                label=label,
                                num_total_classes=num_total_classes,
                                sampler=sampler,
                                custom_dist=word_frequencys,
                                sample_weight=sample_weight,
                                param_attr=fluid.ParamAttr(name=w_param_name),
                                bias_attr=fluid.ParamAttr(name=b_param_name),
                                num_neg_samples=num_neg_samples, is_sparse=is_sparse)

        return cost

    def hsigmoid_layer(input, label, non_leaf_num, max_code_length, data_list):
        hs_cost = None
        ptable = None
        pcode = None
        if max_code_length != None:
            ptable = fluid.layers.data(
                name='ptable', shape=[max_code_length], dtype='int64')
            pcode = fluid.layers.data(
                name='pcode', shape=[max_code_length], dtype='int64')
            data_list.append(pcode)
            data_list.append(ptable)
        else:
            ptable = fluid.layers.data(name='ptable', shape=[40], dtype='int64')
            pcode = fluid.layers.data(name='pcode', shape=[40], dtype='int64')
            data_list.append(pcode)
            data_list.append(ptable)
        if non_leaf_num == None:
            non_leaf_num = dict_size

        cost = fluid.layers.hsigmoid(
            input=input,
            label=label,
            non_leaf_num=non_leaf_num,
            ptable=ptable,
            pcode=pcode,
            is_costum=True)

        return cost

    data_shapes = []
    data_lod_levels = []
    data_types = []

    # input_word
    data_shapes.append((-1, 1))
    data_lod_levels.append(1)
    data_types.append('int64')
    # predict_word
    data_shapes.append((-1, 1))
    data_lod_levels.append(1)
    data_types.append('int64')

    py_reader = fluid.layers.py_reader(capacity=64,
                                       shapes=data_shapes,
                                       lod_levels=data_lod_levels,
                                       dtypes=data_types,
                                       name='py_reader',
                                       use_double_buffer=True)

    word_and_label = fluid.layers.read_file(py_reader)

    cost = None

    emb = fluid.layers.embedding(
        input=word_and_label[0],
        is_sparse=is_sparse,
        size=[dict_size, embedding_size],
        param_attr=fluid.ParamAttr(initializer=fluid.initializer.Normal(
            scale=1 / math.sqrt(dict_size))))

    if with_nce:
        cost = nce_layer(emb, word_and_label[1], embedding_size, dict_size, 5,
                         "uniform", word_frequencys, None)
    if with_hsigmoid:
        cost = hsigmoid_layer(emb, word_and_label[1], dict_size, max_code_length,
                              None)

    avg_cost = fluid.layers.reduce_mean(cost)

    return avg_cost, py_reader
