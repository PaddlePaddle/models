#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
# 
# Copyright (c) 2017 Baidu.com, Inc. All Rights Reserved
# 
########################################################################
"""
File: ngram_language_model.py
Author: chengxiaohua(chengxiaohua@baidu.com)
Date: 2017/05/02 15:13:21
"""
import sqlite3
import paddle.v2 as paddle
import math
import os
import sys
import ptb
import gzip
import numpy as np

batch_size = 1000
word_dim = 256
hidden_dim = 512
lr = 0.1
max_word_num = 40


def gen_embedding_layer(layer_name, voc_dim):
    return paddle.layer.data(
        name=layer_name, type=paddle.data_type.integer_value(voc_dim))


def word_embed(in_layer):
    word_embed = paddle.layer.table_projection(
        input=in_layer,
        size=word_dim,
        param_attr=paddle.attr.Param(
            name="emb", initial_std=0.001, learning_rate=lr, l2_rate=0))
    return word_embed


def ngrm_network(voc_dim):
    firWord = gen_embedding_layer("firWord", voc_dim)
    secWord = gen_embedding_layer("secWord", voc_dim)
    thirdWord = gen_embedding_layer("thirdWord", voc_dim)
    forthWord = gen_embedding_layer("forthWord", voc_dim)
    fifthWord = gen_embedding_layer("fifthWord", voc_dim)

    fir = word_embed(firWord)
    sec = word_embed(secWord)
    third = word_embed(thirdWord)
    forth = word_embed(forthWord)
    fifth = word_embed(firWord)

    embedding = paddle.layer.concat(input=[fir, sec, third, forth, fifth])

    hidden1 = paddle.layer.fc(
        input=embedding,
        act=paddle.activation.Sigmoid(),
        size=hidden_dim,
        bias_attr=paddle.attr.Param(learning_rate=2 * lr),
        layer_attr=paddle.attr.Extra(drop_rate=0.5),
        param_attr=paddle.attr.Param(
            learning_rate=lr, initial_std=1. / math.sqrt(word_dim * 8)))

    output = paddle.layer.fc(
        input=hidden1,
        act=paddle.activation.Softmax(),
        size=voc_dim,
        bias_attr=paddle.attr.Param(learning_rate=2 * lr),
        param_attr=paddle.attr.Param(learning_rate=lr))

    return output


def train_model(word_dict, voc_dim):
    nxtWord = gen_embedding_layer("nxtWord", voc_dim)
    output = ngrm_network(voc_dim)

    def event_handler(event):
        if isinstance(event, paddle.event.EndIteration):
            if event.batch_id % 100 == 0:
                result = trainer.test(
                    paddle.batch(ptb.ngram_test(word_dict, 6), batch_size))
                print("Pass %d, Batch %d, Cost %f, %s, Testing metrics %s" %
                      (event.pass_id, event.batch_id, event.cost, event.metrics,
                       result.metrics))
        if isinstance(event, paddle.event.EndPass):
            model_name = './models/model_pass_%05d.tar.gz' % event.pass_id
            print("Save model into %s !" % model_name)
            with gzip.open(model_name, 'w') as f:
                parameters.to_tar(f)

    cost = paddle.layer.classification_cost(input=output, label=nxtWord)
    parameters = paddle.parameters.create(cost)

    adadelta_optimizer = paddle.optimizer.AdaDelta(
        learning_rate=0.1,
        rho=0.95,
        epsilon=1e-6,
        model_average=paddle.optimizer.ModelAverage(
            average_window=0.5, max_average_window=2500),
        learning_rate_decay_a=0.0,
        learning_rate_decay_b=0.0,
        gradient_clipping_threshold=25)

    trainer = paddle.trainer.SGD(cost, parameters, adadelta_optimizer)
    trainer.train(
        paddle.batch(ptb.ngram_train(word_dict, 6), batch_size),
        num_passes=30,
        event_handler=event_handler)


def predict(word_dict, voc_dim):
    prediction_layer = ngrm_network(voc_dim)

    with gzip.open('./models/model_pass_00000.tar.gz') as f:
        parameters = paddle.parameters.Parameters.from_tar(f)

    idx_word_dict = dict((v, k) for k, v in word_dict.items())
    ins_iter = paddle.dataset.imikolov.test(word_dict, 5)

    for ins in ins_iter():
        gen_seq = []
        cur_predict_word = None
        pre_n_words = [ins[-i] for i in xrange(4, 0, -1)]
        while len(gen_seq) < max_word_num:
            infer_res = paddle.infer(
                output_layer=prediction_layer,
                parameters=parameters,
                input=[pre_n_words])
            sorted_idx_array = np.argsort(-infer_res[0])
            cur_predict_word = sorted_idx_array[0]
            if cur_predict_word == word_dict['<e>']:
                break
            gen_seq.append(cur_predict_word)
            pre_n_words = pre_n_words[-3:] + [cur_predict_word]
        print [idx_word_dict[idx] for idx in gen_seq]


def process(is_generating):
    paddle.init(use_gpu=False, trainer_count=4)
    word_dict = paddle.dataset.imikolov.build_dict()
    voc_dim = len(word_dict)

    if not is_generating:
        train_model(word_dict, voc_dim)
    elif is_generating:
        predict(word_dict, voc_dim)


def usage_helper():
    print "Please execute the command as follows:"
    print "Usage: python ngram_language_model.py --train/generate"
    exit(1)


if __name__ == '__main__':
    if not (len(sys.argv) == 2):
        usage_helper()
    if sys.argv[1] == '--train':
        is_generating = False
    elif sys.argv[1] == '--generate':
        is_generating = True
    else:
        usage_helper()
    process(is_generating)
