#!/usr/bin/env python
#coding=utf-8

import os
import random
import json
import logging

logger = logging.getLogger("paddle")
logger.setLevel(logging.INFO)


def data_reader(data_list, is_train=True):
    """ Data reader.

    Arguments:
        - data_list:  A python list which contains path of training samples.
        - is_train:   A boolean parameter indicating this function is called
                      in training or in inferring.
    """

    def reader():
        """shuffle the data list again at the begining of every pass"""
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
