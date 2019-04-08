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


def skip_gram_word2vec(dict_size, embedding_size, is_sparse=False, neg_num=5):

    datas = []
    input_word = fluid.layers.data(name="input_word", shape=[1], dtype='int64')
    true_word = fluid.layers.data(name='true_label', shape=[1], dtype='int64')
    neg_word = fluid.layers.data(
        name="neg_label", shape=[neg_num], dtype='int64')

    datas.append(input_word)
    datas.append(true_word)
    datas.append(neg_word)

    py_reader = fluid.layers.create_py_reader_by_data(
        capacity=64, feed_list=datas, name='py_reader', use_double_buffer=True)

    words = fluid.layers.read_file(py_reader)
    init_width = 0.5 / embedding_size
    input_emb = fluid.layers.embedding(
        input=words[0],
        is_sparse=is_sparse,
        size=[dict_size, embedding_size],
        param_attr=fluid.ParamAttr(
            name='emb',
            initializer=fluid.initializer.Uniform(-init_width, init_width)))

    true_emb_w = fluid.layers.embedding(
        input=words[1],
        is_sparse=is_sparse,
        size=[dict_size, embedding_size],
        param_attr=fluid.ParamAttr(
            name='emb_w', initializer=fluid.initializer.Constant(value=0.0)))

    true_emb_b = fluid.layers.embedding(
        input=words[1],
        is_sparse=is_sparse,
        size=[dict_size, 1],
        param_attr=fluid.ParamAttr(
            name='emb_b', initializer=fluid.initializer.Constant(value=0.0)))
    neg_word_reshape = fluid.layers.reshape(words[2], shape=[-1, 1])
    neg_word_reshape.stop_gradient = True

    neg_emb_w = fluid.layers.embedding(
        input=neg_word_reshape,
        is_sparse=is_sparse,
        size=[dict_size, embedding_size],
        param_attr=fluid.ParamAttr(
            name='emb_w', learning_rate=1.0))

    neg_emb_w_re = fluid.layers.reshape(
        neg_emb_w, shape=[-1, neg_num, embedding_size])
    neg_emb_b = fluid.layers.embedding(
        input=neg_word_reshape,
        is_sparse=is_sparse,
        size=[dict_size, 1],
        param_attr=fluid.ParamAttr(
            name='emb_b', learning_rate=1.0))

    neg_emb_b_vec = fluid.layers.reshape(neg_emb_b, shape=[-1, neg_num])
    true_logits = fluid.layers.elementwise_add(
        fluid.layers.reduce_sum(
            fluid.layers.elementwise_mul(input_emb, true_emb_w),
            dim=1,
            keep_dim=True),
        true_emb_b)
    input_emb_re = fluid.layers.reshape(
        input_emb, shape=[-1, 1, embedding_size])
    neg_matmul = fluid.layers.matmul(
        input_emb_re, neg_emb_w_re, transpose_y=True)
    neg_matmul_re = fluid.layers.reshape(neg_matmul, shape=[-1, neg_num])
    neg_logits = fluid.layers.elementwise_add(neg_matmul_re, neg_emb_b_vec)
    #nce loss

    label_ones = fluid.layers.fill_constant_batch_size_like(
        true_logits, shape=[-1, 1], value=1.0, dtype='float32')
    label_zeros = fluid.layers.fill_constant_batch_size_like(
        true_logits, shape=[-1, neg_num], value=0.0, dtype='float32')

    true_xent = fluid.layers.sigmoid_cross_entropy_with_logits(true_logits,
                                                               label_ones)
    neg_xent = fluid.layers.sigmoid_cross_entropy_with_logits(neg_logits,
                                                              label_zeros)
    cost = fluid.layers.elementwise_add(
        fluid.layers.reduce_sum(
            true_xent, dim=1),
        fluid.layers.reduce_sum(
            neg_xent, dim=1))
    avg_cost = fluid.layers.reduce_mean(cost)
    return avg_cost, py_reader


def infer_network(vocab_size, emb_size):
    analogy_a = fluid.layers.data(name="analogy_a", shape=[1], dtype='int64')
    analogy_b = fluid.layers.data(name="analogy_b", shape=[1], dtype='int64')
    analogy_c = fluid.layers.data(name="analogy_c", shape=[1], dtype='int64')
    all_label = fluid.layers.data(
        name="all_label",
        shape=[vocab_size, 1],
        dtype='int64',
        append_batch_size=False)
    emb_all_label = fluid.layers.embedding(
        input=all_label, size=[vocab_size, emb_size], param_attr="emb")

    emb_a = fluid.layers.embedding(
        input=analogy_a, size=[vocab_size, emb_size], param_attr="emb")
    emb_b = fluid.layers.embedding(
        input=analogy_b, size=[vocab_size, emb_size], param_attr="emb")
    emb_c = fluid.layers.embedding(
        input=analogy_c, size=[vocab_size, emb_size], param_attr="emb")
    target = fluid.layers.elementwise_add(
        fluid.layers.elementwise_sub(emb_b, emb_a), emb_c)
    emb_all_label_l2 = fluid.layers.l2_normalize(x=emb_all_label, axis=1)
    dist = fluid.layers.matmul(x=target, y=emb_all_label_l2, transpose_y=True)
    values, pred_idx = fluid.layers.topk(input=dist, k=4)
    return values, pred_idx
