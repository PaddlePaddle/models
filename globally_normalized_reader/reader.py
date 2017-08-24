#!/usr/bin/env python
#coding=utf-8
import os
import random
import json
import logging

logger = logging.getLogger("paddle")
logger.setLevel(logging.INFO)


def train_reader(data_list, is_train=True):
    def reader():
        # every pass shuffle the data list again
        if is_train:
            random.shuffle(data_list)

        for train_sample in data_list:
            data = json.load(open(train_sample, "r"))

            start_pos = 0
            doc = []
            same_as_question_word = []
            for l in data['sent_lengths']:
                doc.append(data['context'][start_pos:start_pos + l])
                same_as_question_word.append([
                    [[x]] for x in data['same_as_question_word']
                ][start_pos:start_pos + l])
                start_pos += l

            yield (data['question'], doc, same_as_question_word,
                   data['ans_sentence'], data['ans_start'],
                   data['ans_end'] - data['ans_start'])

    return reader


if __name__ == "__main__":
    from train import choose_samples

    train_list, dev_list = choose_samples("data/featurized")
    for i, item in enumerate(train_reader(train_list)()):
        print(item)
        if i > 5: break
