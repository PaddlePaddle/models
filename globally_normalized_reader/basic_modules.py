#!/usr/bin/env python
#coding=utf-8
import collections

import paddle.v2 as paddle
from paddle.v2.layer import parse_network

__all__ = [
    "stacked_bidirectional_lstm",
    "lstm_by_nested_sequence",
]


def stacked_bidirectional_lstm(inputs, size, depth, drop_rate=0., prefix=""):
    if not isinstance(inputs, collections.Sequence):
        inputs = [inputs]

    lstm_last = []
    for dirt in ["fwd", "bwd"]:
        for i in range(depth):
            input_proj = paddle.layer.mixed(
                name="%s_in_proj_%0d_%s__" % (prefix, i, dirt),
                size=size * 4,
                bias_attr=paddle.attr.Param(initial_std=0.),
                input=[paddle.layer.full_matrix_projection(lstm)] if i else [
                    paddle.layer.full_matrix_projection(in_layer)
                    for in_layer in inputs
                ])
            lstm = paddle.layer.lstmemory(
                input=input_proj,
                bias_attr=paddle.attr.Param(initial_std=0.),
                param_attr=paddle.attr.Param(initial_std=5e-4),
                reverse=(dirt == "bwd"))
        lstm_last.append(lstm)

    final_states = paddle.layer.concat(input=[
        paddle.layer.last_seq(input=lstm_last[0]),
        paddle.layer.first_seq(input=lstm_last[1]),
    ])

    lstm_outs = paddle.layer.concat(
        input=lstm_last,
        layer_attr=paddle.attr.ExtraLayerAttribute(drop_rate=drop_rate))
    return final_states, lstm_outs


def lstm_by_nested_sequence(input_layer, hidden_dim, name="", reverse=False):
    '''
    This is a LSTM implemended by nested recurrent_group.
    Paragraph is a nature nested sequence:
    1. each paragraph is a sequence of sentence.
    2. each sentence is a sequence of words.

    This function ueses the nested recurrent_group to implement LSTM.
    1. The outer group iterates over sentence in a paragraph.
    2. The inner group iterates over words in a sentence.
    3. A LSTM is used to encode sentence, its final outputs is used to
       initialize memory of the LSTM that is used to encode the next sentence.
    4. Parameters are shared among these sentence-encoding LSTMs.
    5. Consequently, this function is just equivalent to concatenate all
       sentences in a paragraph into one (long) sentence, and use one LSTM to
       encode this new long sentence.
    '''

    def lstm_outer_step(lstm_group_input, hidden_dim, reverse, name=''):
        outer_memory = paddle.layer.memory(
            name="__inner_%s_last__" % name, size=hidden_dim)

        def lstm_inner_step(input_layer, hidden_dim, reverse, name):
            inner_memory = paddle.layer.memory(
                name="__inner_state_%s__" % name,
                size=hidden_dim,
                boot_layer=outer_memory)
            input_proj = paddle.layer.fc(size=hidden_dim * 4,
                                         bias_attr=False,
                                         input=input_layer)
            return paddle.networks.lstmemory_unit(
                input=input_proj,
                name="__inner_state_%s__" % name,
                out_memory=inner_memory,
                size=hidden_dim,
                act=paddle.activation.Tanh(),
                gate_act=paddle.activation.Sigmoid(),
                state_act=paddle.activation.Tanh())

        inner_out = paddle.layer.recurrent_group(
            name="__inner_%s__" % name,
            step=lstm_inner_step,
            reverse=reverse,
            input=[lstm_group_input, hidden_dim, reverse, name])

        if reverse:
            inner_last_output = paddle.layer.first_seq(
                input=inner_out,
                name="__inner_%s_last__" % name,
                agg_level=paddle.layer.AggregateLevel.TO_NO_SEQUENCE)
        else:
            inner_last_output = paddle.layer.last_seq(
                input=inner_out,
                name="__inner_%s_last__" % name,
                agg_level=paddle.layer.AggregateLevel.TO_NO_SEQUENCE)
        return inner_out

    return paddle.layer.recurrent_group(
        input=[
            paddle.layer.SubsequenceInput(input_layer), hidden_dim, reverse,
            name
        ],
        step=lstm_outer_step,
        name="__outter_%s__" % name,
        reverse=reverse)


def stacked_bi_lstm_by_nested_seq(input_layer, depth, hidden_dim, prefix=""):
    lstm_final_outs = []
    for dirt in ["fwd", "bwd"]:
        for i in range(depth):
            lstm_out = lstm_by_nested_sequence(
                input_layer=(lstm_out if i else input_layer),
                hidden_dim=hidden_dim,
                name="__%s_%s_%02d__" % (prefix, dirt, i),
                reverse=(dirt == "bwd"))
        lstm_final_outs.append(lstm_out)
    return paddle.layer.concat(input=lstm_final_outs)


if __name__ == "__main__":
    vocab_size = 1024
    emb_dim = 128
    embedding = paddle.layer.embedding(
        input=paddle.layer.data(
            name="word",
            type=paddle.data_type.integer_value_sub_sequence(vocab_size)),
        size=emb_dim)
    print(parse_network(
        stacked_bi_lstm_by_nested_seq(
            input_layer=embedding, depth=3, hidden_dim=128, prefix="__lstm")))
