#!/usr/bin/env python
# -*- coding: utf-8 -*-

import paddle.v2 as paddle
from hsigmoid_conf import network_conf
import gzip


def decode_res(infer_res, dict_size):
    """
    Inferring probabilities are orginized as a complete binary tree.
    The actual labels are leaves (indices are counted from class number).
    This function travels paths decoded from inferring results.
    If the probability >0.5 then go to right child, otherwise go to left child.

    param infer_res: inferring result
    param dict_size: class number
    return predict_lbls: actual class
    """
    predict_lbls = []
    infer_res = infer_res > 0.5
    for i, probs in enumerate(infer_res):
        idx = 0
        result = 1
        while idx < len(probs):
            result <<= 1
            if probs[idx]:
                result |= 1
            if probs[idx]:
                idx = idx * 2 + 2  # right child
            else:
                idx = idx * 2 + 1  # left child

        predict_lbl = result - dict_size
        predict_lbls.append(predict_lbl)
    return predict_lbls


def predict(batch_ins, idx_word_dict, dict_size, prediction_layer, parameters):
    infer_res = paddle.infer(
        output_layer=prediction_layer, parameters=parameters, input=batch_ins)

    predict_lbls = decode_res(infer_res, dict_size)
    predict_words = [idx_word_dict[lbl] for lbl in predict_lbls]  # map to word

    # Ouput format: word1 word2 word3 word4 -> predict label
    for i, ins in enumerate(batch_ins):
        print(idx_word_dict[ins[0]] + ' ' + \
            idx_word_dict[ins[1]] + ' ' + \
            idx_word_dict[ins[2]] + ' ' + \
            idx_word_dict[ins[3]] + ' ' + \
         ' -> ' + predict_words[i])


def main():
    paddle.init(use_gpu=False, trainer_count=1)
    word_dict = paddle.dataset.imikolov.build_dict(min_word_freq=2)
    dict_size = len(word_dict)
    prediction_layer = network_conf(
        is_train=False, hidden_size=256, embed_size=32, dict_size=dict_size)

    with gzip.open('./models/model_pass_00000.tar.gz') as f:
        parameters = paddle.parameters.Parameters.from_tar(f)

    idx_word_dict = dict((v, k) for k, v in word_dict.items())
    batch_size = 64
    batch_ins = []
    ins_iter = paddle.dataset.imikolov.test(word_dict, 5)

    for ins in ins_iter():
        batch_ins.append(ins[:-1])
        if len(batch_ins) == batch_size:
            predict(batch_ins, idx_word_dict, dict_size, prediction_layer,
                    parameters)
            batch_ins = []

    if len(batch_ins) > 0:
        predict(batch_ins, idx_word_dict, dict_size, prediction_layer,
                parameters)


if __name__ == '__main__':
    main()
