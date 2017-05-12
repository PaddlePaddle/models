#!/usr/bin/env python
# -*- coding: utf-8 -*-
import paddle.v2 as paddle
import math
import os
import sys
import gzip
import numpy as np

batch_size = 1000
word_dim = 256
hidden_dim = 512
lr = 0.1
max_word_num = 40


def gen_word_embedding(layer_name, voc_dim):
    word_id = paddle.layer.data(
        layer_name, type=paddle.data_type.integer_value(voc_dim))
    word_embedding = paddle.layer.embedding(
        input=word_id,
        size=voc_dim,
        param_attr=paddle.attr.Param(
            name="emb", initial_std=0.001, learning_rate=lr, l2_rate=0))

    return word_embedding


def ngram_network(voc_dim):
    first_embedding = gen_word_embedding("first_word", voc_dim)
    second_embedding = gen_word_embedding("second_word", voc_dim)
    third_embedding = gen_word_embedding("third_word", voc_dim)
    forth_embedding = gen_word_embedding("forth_word", voc_dim)
    fifth_embedding = gen_word_embedding("fifth_word", voc_dim)

    embedding = paddle.layer.concat(input=[
        first_embedding, second_embedding, third_embedding, forth_embedding,
        fifth_embedding
    ])

    hidden1 = paddle.layer.fc(
        input=embedding,
        act=paddle.activation.Tanh(),
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
    #treate the next_word as label
    next_word = paddle.layer.data(
        name="next_word", type=paddle.data_type.integer_value(voc_dim))
    #construct ngram network
    output = ngram_network(voc_dim)

    #callback function when meet some conditions
    def event_handler(event):
        if isinstance(event, paddle.event.EndIteration):
            if event.batch_id % 100 == 0:
                result = trainer.test(
                    paddle.batch(
                        paddle.dataset.imikolov.test(word_dict, 6), batch_size))
                print("Pass %d, Batch %d, Cost %f, %s, Testing metrics %s" %
                      (event.pass_id, event.batch_id, event.cost, event.metrics,
                       result.metrics))
        if isinstance(event, paddle.event.EndPass):
            model_name = './models/model_pass_%05d.tar.gz' % event.pass_id
            print("Save model into %s !" % model_name)
            with gzip.open(model_name, 'w') as f:
                parameters.to_tar(f)

    #calc train loss
    cost = paddle.layer.classification_cost(input=output, label=next_word)
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
        paddle.batch(paddle.dataset.imikolov.train(word_dict, 6), batch_size),
        num_passes=30,
        event_handler=event_handler)


def generate_sequence(word_dict, voc_dim):
    """
    the language model is given N words, the N+1 word should be predicted.if 
    complement language model by ngram, firstly the N words should be embedding seperately;
    secondly concating the N words's embedding as an input is fully connected ,and the 
    output is an embedding; thirdly apply a multiple classifier to predict the probability
    indicating the possibility of the N+1 word for all of the words in word dict;
    In this example, one-way search is used to select the highest probability of the N+1 word
    , I suggest authors could use beam-search to do it.
    At last, the generating process goes as follows:
      1 load the trained model
      2 read the testing data by ins_iter which is a five-element couple.
        according to language model, we can use the five words as N words, and predict
        the following words until the word count is moren than max_word_num or the word of '<e>'
        is selected
      3 so far, we have completed one five-element couple predicting,
        and then continue to iterate the next couple
    """
    prediction_layer = ngram_network(voc_dim)

    with gzip.open('./models/model_pass_00000.tar.gz') as f:
        parameters = paddle.parameters.Parameters.from_tar(f)

    idx_word_dict = dict((v, k) for k, v in word_dict.items())
    ins_iter = paddle.dataset.imikolov.test(word_dict, 5)

    for ins in ins_iter():
        gen_seq = []
        cur_predict_word = None
        pre_n_words = [ins[-i] for i in xrange(5, 0, -1)]
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
            pre_n_words = pre_n_words[-4:] + [cur_predict_word]
        print [idx_word_dict[idx] for idx in gen_seq]


def process(is_generating):
    paddle.init(use_gpu=False, trainer_count=1)
    word_dict = paddle.dataset.imikolov.build_dict()
    voc_dim = len(word_dict)

    if not is_generating:
        train_model(word_dict, voc_dim)
    elif is_generating:
        generate_sequence(word_dict, voc_dim)


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
