#!/usr/bin/env python
# -*- encoding:utf-8 -*-
import math
import paddle.v2 as paddle


def ngram_lm(hidden_size, emb_size, dict_size, gram_num=4, is_train=True):
    emb_layers = []
    embed_param_attr = paddle.attr.Param(
        name="_proj", initial_std=0.001, learning_rate=1, l2_rate=0)
    for i in range(gram_num):
        word = paddle.layer.data(
            name="__word%02d__" % (i),
            type=paddle.data_type.integer_value(dict_size))
        emb_layers.append(
            paddle.layer.embedding(
                input=word, size=emb_size, param_attr=embed_param_attr))

    next_word = paddle.layer.data(
        name="__target_word__", type=paddle.data_type.integer_value(dict_size))

    context_embedding = paddle.layer.concat(input=emb_layers)

    hidden_layer = paddle.layer.fc(
        input=context_embedding,
        size=hidden_size,
        act=paddle.activation.Tanh(),
        param_attr=paddle.attr.Param(initial_std=1. / math.sqrt(emb_size * 8)))

    if is_train:
        cost = paddle.layer.nce(
            input=hidden_layer,
            label=next_word,
            num_classes=dict_size,
            param_attr=paddle.attr.Param(name="nce_w"),
            bias_attr=paddle.attr.Param(name="nce_b"),
            act=paddle.activation.Sigmoid(),
            num_neg_samples=25,
            neg_distribution=None)
        return cost
    else:
        prediction = paddle.layer.fc(
            size=dict_size,
            act=paddle.activation.Softmax(),
            bias_attr=paddle.attr.Param(name="nce_b"),
            input=hidden_layer,
            param_attr=paddle.attr.Param(name="nce_w"))

        return prediction
