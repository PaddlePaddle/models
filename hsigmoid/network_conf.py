#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import paddle.v2 as paddle


def ngram_lm(hidden_size, embed_size, dict_size, gram_num=4, is_train=True):
    emb_layers = []
    embed_param_attr = paddle.attr.Param(
        name="_proj", initial_std=0.001, learning_rate=1, l2_rate=0)
    for i in range(gram_num):
        word = paddle.layer.data(
            name="__word%02d__" % (i),
            type=paddle.data_type.integer_value(dict_size))
        emb_layers.append(
            paddle.layer.embedding(
                input=word, size=embed_size, param_attr=embed_param_attr))

    target_word = paddle.layer.data(
        name="__target_word__", type=paddle.data_type.integer_value(dict_size))

    embed_context = paddle.layer.concat(input=emb_layers)

    hidden_layer = paddle.layer.fc(
        input=embed_context,
        size=hidden_size,
        act=paddle.activation.Sigmoid(),
        layer_attr=paddle.attr.Extra(drop_rate=0.5),
        bias_attr=paddle.attr.Param(learning_rate=2),
        param_attr=paddle.attr.Param(
            initial_std=1. / math.sqrt(embed_size * 8), learning_rate=1))

    if is_train == True:
        cost = paddle.layer.hsigmoid(
            input=hidden_layer,
            label=target_word,
            num_classes=dict_size,
            param_attr=paddle.attr.Param(name="sigmoid_w"),
            bias_attr=paddle.attr.Param(name="sigmoid_b"))
        return cost
    else:
        prediction = paddle.layer.fc(
            size=dict_size - 1,
            input=hidden_layer,
            act=paddle.activation.Sigmoid(),
            bias_attr=paddle.attr.Param(name="sigmoid_b"),
            param_attr=paddle.attr.Param(name="sigmoid_w"))
        return prediction
