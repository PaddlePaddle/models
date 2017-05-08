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
Conll03 dataset.
"""

import tarfile
import gzip
import itertools
import re
import numpy as np

__all__ = ['train', 'test', 'get_dict', 'get_embedding']

UNK_IDX = 0


def canonicalize_digits(word):
    if any([c.isalpha() for c in word]): return word
    word = re.sub("\d", "DG", word)
    if word.startswith("DG"):
        word = word.replace(",", "")  # remove thousands separator
    return word


def canonicalize_word(word, wordset=None, digits=True):
    word = word.lower()
    if digits:
        if (wordset != None) and (word in wordset): return word
        word = canonicalize_digits(word)  # try to canonicalize numbers
    if (wordset == None) or (word in wordset): return word
    else: return "UUUNKKK"  # unknown token


def load_dict(filename):
    d = dict()
    with open(filename, 'r') as f:
        for i, line in enumerate(f):
            d[line.strip()] = i
    return d


def get_dict(vocab_file='data/vocab.txt', target_file='data/target.txt'):
    """
    Get the word and label dictionary.
    """
    word_dict = load_dict(vocab_file)
    label_dict = load_dict(target_file)
    return word_dict, label_dict


def get_embedding(emb_file='data/wordVectors.txt'):
    """
    Get the trained word vector.
    """
    return np.loadtxt(emb_file, dtype=float)


def corpus_reader(filename='data/train'):
    def reader():
        sentence = []
        labels = []
        with open(filename) as f:
            for line in f:
                if re.match(r"-DOCSTART-.+", line) or (len(line.strip()) == 0):
                    if len(sentence) > 0:
                        yield sentence, labels
                    sentence = []
                    labels = []
                else:
                    segs = line.strip().split()
                    sentence.append(segs[0])
                    labels.append(segs[-1])

        f.close()

    return reader


def reader_creator(corpus_reader=corpus_reader('data/train'),
                   word_dict=load_dict('data/vocab.txt'),
                   label_dict=load_dict('data/target.txt')):
    """
    Conll03 train set creator.

    Because the training dataset is not free, the test dataset is used for
    training. It returns a reader creator, each sample in the reader is nine
    features, including sentence sequence, predicate, predicate context,
    predicate context flag and tagged sequence.

    :return: Training reader creator
    :rtype: callable
    """

    def reader():
        for sentence, labels in corpus_reader():
            #word_idx = [word_dict.get(w, UNK_IDX) for w in sentence]
            word_idx = [
                word_dict.get(canonicalize_word(w, word_dict), UNK_IDX)
                for w in sentence
            ]
            label_idx = [label_dict.get(w) for w in labels]
            yield word_idx, label_idx

    return reader


def train():
    return reader_creator(
        corpus_reader('data/train'),
        word_dict=load_dict('data/vocab.txt'),
        label_dict=load_dict('data/target.txt'))


def test():
    return reader_creator(
        corpus_reader('data/test'),
        word_dict=load_dict('data/vocab.txt'),
        label_dict=load_dict('data/target.txt'))
