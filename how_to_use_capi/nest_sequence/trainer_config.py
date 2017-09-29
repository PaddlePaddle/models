#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys

import paddle.v2 as paddle
from paddle.trainer_config_helpers import *

dict_dim = 1000
hidden_size = 128
label_dim = 3

data = paddle.layer.data("text",
                         paddle.data_type.integer_value_sub_sequence(dict_dim))
emb = paddle.layer.embedding(input=data, size=128)


def lstm_group(lstm_group_input):
    group_input = paddle.layer.mixed(
        size=hidden_size * 4,
        input=[paddle.layer.full_matrix_projection(input=lstm_group_input)])
    lstm_output = paddle.networks.lstmemory_group(
        input=group_input,
        name="lstm_group",
        size=hidden_size,
        act=paddle.activation.Tanh(),
        gate_act=paddle.activation.Sigmoid(),
        state_act=paddle.activation.Tanh())
    return lstm_output


lstm_nest_group = paddle.layer.recurrent_group(
    input=paddle.layer.SubsequenceInput(emb),
    step=lstm_group,
    name="lstm_nest_group")

lstm_last = paddle.layer.last_seq(
    input=lstm_nest_group, agg_level=paddle.layer.AggregateLevel.TO_SEQUENCE)

lstm_average = paddle.layer.pooling(
    input=lstm_last,
    pooling_type=paddle.pooling.Avg(),
    agg_level=paddle.layer.AggregateLevel.TO_NO_SEQUENCE)

output = paddle.layer.mixed(
    size=label_dim,
    input=[paddle.layer.full_matrix_projection(input=lstm_average)],
    act=paddle.activation.Softmax(),
    bias_attr=True)

outputs(output)
