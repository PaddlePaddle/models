#!/usr/bin/env python
# coding=utf-8
import os
import logging
from collections import defaultdict

__all__ = ["build_dict", "load_dict"]

logger = logging.getLogger("paddle")
logger.setLevel(logging.DEBUG)


def build_dict(data_file,
               save_path,
               max_word_num,
               cutoff_word_fre=5,
               insert_extra_words=["<unk>", "<e>"]):
    """
    :param data_file: path of data file
    :param save_path: path to save the word dictionary
    :param vocab_max_size: if vocab_max_size is set, top vocab_max_size words
        will be added into word vocabulary
    :param cutoff_thd: if cutoff_thd is set, words whose frequencies are less
        than cutoff_thd will not added into word vocabulary.
        NOTE that: vocab_max_size and cutoff_thd cannot be set at the same time
    :param extra_keys: extra keys defined by users that added into the word
        dictionary, ususally these keys includes <unk>, start and ending marks
    """
    word_count = defaultdict(int)
    with open(data_file, "r") as f:
        for idx, line in enumerate(f):
            if not (idx + 1) % 100000:
                logger.debug("processing %d lines ... " % (idx + 1))
            words = line.strip().lower().split()
            for w in words:
                word_count[w] += 1

    sorted_words = sorted(
        word_count.iteritems(), key=lambda x: x[1], reverse=True)

    stop_pos = len(sorted_words) if sorted_words[-1][
        1] > cutoff_word_fre else next(idx for idx, v in enumerate(sorted_words)
                                       if v[1] < cutoff_word_fre)

    stop_pos = min(max_word_num, stop_pos)
    with open(save_path, "w") as fdict:
        for w in insert_extra_words:
            fdict.write("%s\t-1\n" % (w))
        for idx, info in enumerate(sorted_words):
            if idx == stop_pos: break
            fdict.write("%s\t%d\n" % (info[0], info[-1]))


def load_dict(dict_path):
    """
    :param dict_path: path of word dictionary
    """
    return dict((line.strip().split("\t")[0], idx)
                for idx, line in enumerate(open(dict_path, "r").readlines()))


def load_reverse_dict(dict_path):
    return dict((idx, line.strip().split("\t")[0])
                for idx, line in enumerate(open(dict_path, "r").readlines()))
