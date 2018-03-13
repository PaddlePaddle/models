#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging

import paddle.fluid as fluid

import numpy as np

logger = logging.getLogger("paddle")
logger.setLevel(logging.INFO)


def get_embedding(emb_file='data/wordVectors.txt'):
    """
    Get the trained word vector.
    """
    return np.loadtxt(emb_file, dtype='float32')


def load_dict(dict_path):
    """
    Load the word dictionary from the given file.
    Each line of the given file is a word, which can include multiple columns
    seperated by tab.

    This function takes the first column (columns in a line are seperated by
    tab) as key and takes line number of a line as the key (index of the word
    in the dictionary).
    """

    return dict((line.strip().split("\t")[0], idx)
                for idx, line in enumerate(open(dict_path, "r").readlines()))


def load_reverse_dict(dict_path):
    """
    Load the word dictionary from the given file.
    Each line of the given file is a word, which can include multiple columns
    seperated by tab.

    This function takes line number of a line as the key (index of the word in
    the dictionary) and the first column (columns in a line are seperated by
    tab) as the value.
    """
    return dict((idx, line.strip().split("\t")[0])
                for idx, line in enumerate(open(dict_path, "r").readlines()))


def to_lodtensor(data, place):
    seq_lens = [len(seq) for seq in data]
    cur_len = 0
    lod = [cur_len]
    for l in seq_lens:
        cur_len += l
        lod.append(cur_len)
    flattened_data = np.concatenate(data, axis=0).astype("int64")
    flattened_data = flattened_data.reshape([len(flattened_data), 1])
    res = fluid.LoDTensor()
    res.set(flattened_data, place)
    res.set_lod([lod])
    return res
