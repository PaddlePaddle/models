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
    :type data_file: str
    :param save_path: path to save the word dictionary
    :type save_path: str
    :param vocab_max_size: if vocab_max_size is set, top vocab_max_size words
        will be added into word vocabulary
    :type vocab_max_size: int
    :param cutoff_thd: if cutoff_thd is set, words whose frequencies are less
        than cutoff_thd will not be added into word vocabulary.
        NOTE that: vocab_max_size and cutoff_thd cannot be set at the same time
    :type cutoff_word_fre: int
    :param extra_keys: extra keys defined by users that added into the word
        dictionary, ususally these keys include <unk>, start and ending marks
    :type extra_keys: list
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
    load word dictionary from the given file. Each line of the give file is
    a word in the word dictionary. The first column of the line, seperated by
    TAB, is the key, while the line index is the value.

    :param dict_path: path of word dictionary
    :type dict_path: str
    :return: the dictionary
    :rtype: dict
    """
    return dict((line.strip().split("\t")[0], idx)
                for idx, line in enumerate(open(dict_path, "r").readlines()))


def load_reverse_dict(dict_path):
    """
    load word dictionary from the given file. Each line of the give file is
    a word in the word dictionary. The line index is the key, while the first
    column of the line, seperated by TAB, is the value.

    :param dict_path: path of word dictionary
    :type dict_path: str
    :return: the dictionary
    :rtype: dict
    """
    return dict((idx, line.strip().split("\t")[0])
                for idx, line in enumerate(open(dict_path, "r").readlines()))
