#!/usr/bin/env python
#coding=utf-8
import pdb
import os
import random
import json


def train_reader(data_list, is_train=True):
    def reader():
        # every pass shuffle the data list again
        if is_train:
            random.shuffle(data_list)

        for train_sample in data_list:
            data = json.load(open(train_sample, "r"))
            sent_len = data['sent_lengths']

            doc_len = len(data['context'])
            same_as_question_word = [[[x]]
                                     for x in data['same_as_question_word']]

            ans_sentence = [0] * doc_len
            ans_sentence[data['ans_sentence']] = 1

            ans_start = [0] * doc_len
            ans_start[data['ans_start']] = 1

            ans_end = [0] * doc_len
            ans_end[data['ans_end']] = 1
            yield (data['question'], data['context'], same_as_question_word,
                   ans_sentence, ans_start, ans_end)

    return reader


if __name__ == "__main__":
    from train import choose_samples

    train_list, dev_list = choose_samples("data/featurized")
    for i, item in enumerate(train_reader(train_list)()):
        print(item)
        if i > 5: break
