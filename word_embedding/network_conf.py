#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import paddle.v2 as paddle


def network_conf(hidden_size, embed_size, dict_size):
    def word_embed(in_layer):
        ''' word embedding layer '''
        word_embed = paddle.layer.table_projection(
            input=in_layer,
            size=embed_size,
            param_attr=paddle.attr.Param(
                name="_proj", initial_std=0.001, learning_rate=1, l2_rate=0))
        return word_embed

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

    first_word_embed = word_embed(first_word)
    second_word_embed = word_embed(second_word)
    third_word_embed = word_embed(third_word)
    fourth_word_embed = word_embed(fourth_word)

    context_embed = paddle.layer.concat(input=[
        first_word_embed, second_word_embed, third_word_embed, fourth_word_embed
    ])

    hidden_layer = paddle.layer.fc(
        input=context_embed,
        size=hidden_size,
        act=paddle.activation.Sigmoid(),
        layer_attr=paddle.attr.Extra(drop_rate=0.5),
        bias_attr=paddle.attr.Param(learning_rate=2),
        param_attr=paddle.attr.Param(
            initial_std=1. / math.sqrt(embed_size * 8), learning_rate=1))

    cost = paddle.layer.hsigmoid(
        input=hidden_layer,
        label=target_word,
        num_classes=dict_size,
        param_attr=paddle.attr.Param(name='sigmoid_w'),
        bias_attr=paddle.attr.Param(name='sigmoid_b'))

    with paddle.layer.mixed(
            size=dict_size - 1,
            act=paddle.activation.Sigmoid(),
            bias_attr=paddle.attr.Param(name='sigmoid_b')) as prediction:
        prediction += paddle.layer.trans_full_matrix_projection(
            input=hidden_layer, param_attr=paddle.attr.Param(name='sigmoid_w'))

    input_data_lst = ['firstw', 'secondw', 'thirdw', 'fourthw', 'fifthw']

    return input_data_lst, cost, prediction
