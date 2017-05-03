#!/usr/bin/env python
# -*- coding: utf-8 -*-

import paddle.v2 as paddle
from network_conf import network_conf
import gzip


def decode_res(infer_res, dict_size):
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


def main():
    paddle.init(use_gpu=False, trainer_count=4)
    word_dict = paddle.dataset.imikolov.build_dict()
    dict_size = len(word_dict)
    _, _, prediction = network_conf(
        hidden_size=256, embed_size=32, dict_size=dict_size)

    print('Load model ....')
    with gzip.open('./models/model_pass_00000.tar.gz') as f:
        parameters = paddle.parameters.Parameters.from_tar(f)

        ins_num = 10
        ins_lst = []
        ins_lbls = []

        ins_buffer = paddle.reader.shuffle(
            lambda: paddle.dataset.imikolov.train(word_dict, 5)(),
            buf_size=1000)

        for ins in ins_buffer():
            ins_lst.append(ins[:-1])
            ins_lbls.append(ins[-1])
            if len(ins_lst) >= ins_num: break

        infer_res = paddle.infer(
            output_layer=prediction, parameters=parameters, input=ins_lst)

        idx_word_dict = dict((v, k) for k, v in word_dict.items())

        predict_lbls = decode_res(infer_res, dict_size)
        predict_words = [idx_word_dict[lbl] for lbl in predict_lbls]
        gt_words = [idx_word_dict[lbl] for lbl in ins_lbls]

        for i, ins in enumerate(ins_lst):
            print idx_word_dict[ins[0]] + ' ' + idx_word_dict[ins[1]] + \
             ' -> ' + predict_words[i] + ' ( ' + gt_words[i] + ' )'


if __name__ == '__main__':
    main()
