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

    words = []
    input_word = fluid.data(name="input_word", shape=[None, 1], dtype='int64')
    true_word = fluid.data(name='true_label', shape=[None, 1], dtype='int64')
    neg_word = fluid.data(
        name="neg_label", shape=[None, neg_num], dtype='int64')

    words.append(input_word)
    words.append(true_word)
    words.append(neg_word)

    data_loader = fluid.io.DataLoader.from_generator(
        capacity=64, feed_list=words, iterable=False)

    init_width = 0.5 / embedding_size
    input_emb = fluid.embedding(
        input=words[0],
        is_sparse=is_sparse,
        size=[dict_size, embedding_size],
        param_attr=fluid.ParamAttr(
            name='emb',
            initializer=fluid.initializer.Uniform(-init_width, init_width)))

    true_emb_w = fluid.embedding(
        input=words[1],
        is_sparse=is_sparse,
        size=[dict_size, embedding_size],
        param_attr=fluid.ParamAttr(
            name='emb_w', initializer=fluid.initializer.Constant(value=0.0)))

    true_emb_b = fluid.embedding(
        input=words[1],
        is_sparse=is_sparse,
        size=[dict_size, 1],
        param_attr=fluid.ParamAttr(
            name='emb_b', initializer=fluid.initializer.Constant(value=0.0)))
    input_emb = fluid.layers.squeeze(input=input_emb, axes=[1])
    true_emb_w = fluid.layers.squeeze(input=true_emb_w, axes=[1])
    true_emb_b = fluid.layers.squeeze(input=true_emb_b, axes=[1])

    neg_emb_w = fluid.embedding(
        input=words[2],
        is_sparse=is_sparse,
        size=[dict_size, embedding_size],
        param_attr=fluid.ParamAttr(
            name='emb_w', learning_rate=1.0))
    neg_emb_b = fluid.embedding(
        input=words[2],
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
    neg_matmul = fluid.layers.matmul(input_emb_re, neg_emb_w, transpose_y=True)
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
    return avg_cost, data_loader


def infer_network(vocab_size, emb_size):
    analogy_a = fluid.data(name="analogy_a", shape=[None], dtype='int64')
    analogy_b = fluid.data(name="analogy_b", shape=[None], dtype='int64')
    analogy_c = fluid.data(name="analogy_c", shape=[None], dtype='int64')
    all_label = fluid.data(name="all_label", shape=[vocab_size], dtype='int64')
    emb_all_label = fluid.embedding(
        input=all_label, size=[vocab_size, emb_size], param_attr="emb")

    emb_a = fluid.embedding(
        input=analogy_a, size=[vocab_size, emb_size], param_attr="emb")
    emb_b = fluid.embedding(
        input=analogy_b, size=[vocab_size, emb_size], param_attr="emb")
    emb_c = fluid.embedding(
        input=analogy_c, size=[vocab_size, emb_size], param_attr="emb")
    target = fluid.layers.elementwise_add(
        fluid.layers.elementwise_sub(emb_b, emb_a), emb_c)
    emb_all_label_l2 = fluid.layers.l2_normalize(x=emb_all_label, axis=1)
    dist = fluid.layers.matmul(x=target, y=emb_all_label_l2, transpose_y=True)
    values, pred_idx = fluid.layers.topk(input=dist, k=4)
    return values, pred_idx
