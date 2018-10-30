# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
"""

import paddle.dataset.common
import collections
import tarfile
import re
import string
import random
import os, sys
import nltk
from os.path import expanduser


__all__ = ['word_dict', 'train', 'dev', 'test']

URL = "default_url"

DATA_HOME = os.path.expanduser('~/.cache/paddle/dataset/')
DATA_DIR = "QQP"

QUORA_TRAIN_FILE_NAME = os.path.join(DATA_HOME, DATA_DIR, 'train.tsv')
QUORA_DEV_FILE_NAME = os.path.join(DATA_HOME, DATA_DIR, 'dev.tsv')
QUORA_TEST_FILE_NAME = os.path.join(DATA_HOME, DATA_DIR, 'test.tsv')

# punctuation or nltk or space
TOKENIZE_METHOD='nltk'

def tokenize(s):
    if sys.version_info <= (3, 0): # for python2
        s = s.decode('utf-8')
    if TOKENIZE_METHOD == "nltk":
        return nltk.tokenize.word_tokenize(s)
    elif TOKENIZE_METHOD == "punctuation":
        return s.translate({ord(char): None for char in string.punctuation}).lower().split()
    elif TOKENIZE_METHOD == "space":
        return s.split()
    else:
        raise RuntimeError("Invalid tokenize method")

def maybe_open(file_name):
    if not os.path.isfile(file_name):
        msg = "file not exist: %s\nPlease download the dataset firstly from: %s\n\n" % (file_name, URL) + \
                ("# The finally dataset dir should be like\n\n"
                "$HOME/.cache/paddle/dataset\n"
                " |- QQP\n"
                "     |- train.tsv\n"
                "     |- test.tsv\n"
                "     |- dev.tsv\n"
                "     |- original\n"
                "         |- quora_duplicate_questions.tsv\n")
        raise RuntimeError(msg)

    return open(file_name, 'r')

def tokenized_question_pairs_without_label(file_name):
    """
    This is for test question pairs which have no labels
    """
    COLUMN_COUNT = 3 
    with maybe_open(file_name) as f:
        lines = f.readlines()
        first_line = True
        for line in lines:
            if first_line == True:
                first_line = False
                continue
            info = line.strip('\n').split('\t')
            if len(info) != COLUMN_COUNT:
                # formatting error
                continue
            (id, question1, question2) = info
            question1 = tokenize(question1)
            question2 = tokenize(question2)
            # [] is not allowed in fluid
            if question1 == []: question1 = ['_']
            if question2 == []: question2 = ['_']
            yield question1, question2


def tokenized_question_pairs_with_label(file_name):
    """
    This is for train and test question pairs with labels
    """
    COLUMN_COUNT = 6
    with maybe_open(file_name) as f:
        lines = f.readlines()
        first_line = True
        for line in lines:
            if first_line == True:
                first_line = False
                continue
            info = line.strip('\n').split('\t')
            if len(info) != COLUMN_COUNT:
                # formatting error
                continue
            (id, qid1, qid2, question1, question2, label) = info
            question1 = tokenize(question1)
            question2 = tokenize(question2)
            # [] is not allowed in fluid
            if question1 == []: question1 = ['_']
            if question2 == []: question2 = ['_']
            yield question1, question2, int(label)


def tokenized_questions(file_name):
    """
    yield all questions for generating word2id dict
    """
    COLUMN_COUNT = 6
    with maybe_open(file_name) as f:
        lines = f.readlines()
        first_line = True
        for line in lines:
            if first_line == True:
                first_line = False
                continue
            info = line.strip('\n').split('\t')
            if len(info) != COLUMN_COUNT:
                # formatting error
                continue
            (id, qid1, qid2, question1, question2, label) = info
            yield tokenize(question1)
            yield tokenize(question2)


def build_dict(file_name, cutoff):
    """
    Build a word dictionary from the corpus. Keys of the dictionary are words,
    and values are zero-based IDs of these words.
    """
    word_freq = collections.defaultdict(int)
    for doc in tokenized_questions(file_name):
        for word in doc:
            word_freq[word] += 1

    word_freq = filter(lambda x: x[1] > cutoff, word_freq.items())

    dictionary = sorted(word_freq, key=lambda x: (-x[1], x[0]))
    words, _ = list(zip(*dictionary))
    word_idx = dict(zip(words, range(len(words))))
    word_idx['<unk>'] = len(words)
    word_idx['<pad>'] = len(words) + 1
    return word_idx


def reader_creator(file_name, word_idx, phase):
    UNK_ID = word_idx['<unk>']

    if phase == 'train' or phase == 'dev':
        def reader():
            for (q1, q2, label) in tokenized_question_pairs_with_label(file_name):
                q1_ids = [word_idx.get(w, UNK_ID) for w in q1]
                q2_ids = [word_idx.get(w, UNK_ID) for w in q2]
                if q1_ids != [] and q2_ids != []: # [] is not allowed in fluid
                    assert(label in [0, 1])
                    yield q1_ids, q2_ids, label    
        return reader

    else: # phase == test
        def reader():
            for (q1, q2) in tokenized_question_pairs_without_label(file_name):
                q1_ids = [word_idx.get(w, UNK_ID) for w in q1]
                q2_ids = [word_idx.get(w, UNK_ID) for w in q2]
                if q1_ids != [] and q2_ids != []: # [] is not allowed in fluid
                    yield q1_ids, q2_ids
        return reader


def train(word_idx):
    """
    Quora training set creator.

    It returns a reader creator, each sample in the reader is two zero-based ID
    list and label in [0, 1].

    :param word_idx: word dictionary
    :type word_idx: dict
    :return: Training reader creator
    :rtype: callable
    """   
    return reader_creator(QUORA_TRAIN_FILE_NAME, word_idx, phase='train')


def dev(word_idx):
    """
    Quora develop set creator.

    It returns a reader creator, each sample in the reader is two zero-based ID
    list and label in [0, 1].

    :param word_idx: word dictionary
    :type word_idx: dict
    :return: develop reader creator
    :rtype: callable

    """
    return reader_creator(QUORA_DEV_FILE_NAME, word_idx, phase='dev')

def test(word_idx):
    """
    Quora test set creator.

    It returns a reader creator, each sample in the reader is two zero-based ID
    list and label in [0, 1].

    :param word_idx: word dictionary
    :type word_idx: dict
    :return: Test reader creator
    :rtype: callable
    """
    return reader_creator(QUORA_TEST_FILE_NAME, word_idx, phase='test')

def word_dict():
    """
    Build a word dictionary from the corpus.

    :return: Word dictionary
    :rtype: dict
    """
    return build_dict(file_name=QUORA_TRAIN_FILE_NAME, cutoff=4)

