#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import paddle.v2 as paddle


def network_conf(hidden_size, embed_size, dict_size, is_train=True):
    first_word = paddle.layer.data(
        name='firstw', type=paddle.data_type.integer_value(dict_size))
    second_word = paddle.layer.data(
        name='secondw', type=paddle.data_type.integer_value(dict_size))
    third_word = paddle.layer.data(
        name='thirdw', type=paddle.data_type.integer_value(dict_size))
    fourth_word = paddle.layer.data(
        name='fourthw', type=paddle.data_type.integer_value(dict_size))
    target_word = paddle.layer.data(
        name='fifthw', type=paddle.data_type.integer_value(dict_size))

    embed_param_attr = paddle.attr.Param(
        name="_proj", initial_std=0.001, learning_rate=1, l2_rate=0)
    embed_first_word = paddle.layer.embedding(
        input=first_word, size=embed_size, param_attr=embed_param_attr)
    embed_second_word = paddle.layer.embedding(
        input=second_word, size=embed_size, param_attr=embed_param_attr)
    embed_third_word = paddle.layer.embedding(
        input=third_word, size=embed_size, param_attr=embed_param_attr)
    embed_fourth_word = paddle.layer.embedding(
        input=fourth_word, size=embed_size, param_attr=embed_param_attr)

    embed_context = paddle.layer.concat(input=[
        embed_first_word, embed_second_word, embed_third_word, embed_fourth_word
    ])

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
            param_attr=paddle.attr.Param(name='sigmoid_w'),
            bias_attr=paddle.attr.Param(name='sigmoid_b'))
        return cost
    else:
        with paddle.layer.mixed(
                size=dict_size - 1,
                act=paddle.activation.Sigmoid(),
                bias_attr=paddle.attr.Param(name='sigmoid_b')) as prediction:
            prediction += paddle.layer.trans_full_matrix_projection(
                input=hidden_layer,
                param_attr=paddle.attr.Param(name='sigmoid_w'))
        return prediction
