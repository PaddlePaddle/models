# coding=utf-8
import pdb
import collections
import os

MIN_LEN = 3
MAX_LEN = 100


def rnn_reader(file_name, word_dict, is_infer):
    """
    create reader for RNN, each line is a sample.

    :param file_name: file name.
    :param min_sentence_length: sentence's min length.
    :param max_sentence_length: sentence's max length.
    :param word_dict: vocab with content of '{word, id}',
                      'word' is string type , 'id' is int type.
    :return: data reader.
    """

    def reader():
        UNK_ID = word_dict['<unk>']
        with open(file_name) as file:
            for line in file:
                words = line.strip().split()
                if len(words) < MIN_LEN or len(words) > MAX_LEN:
                    continue
                ids = [word_dict.get(w, UNK_ID)
                       for w in words] + [word_dict['<e>']] * 2
                if is_infer:
                    yield ids[:-1]
                else:
                    yield ids[:-1], ids[1:]

    return reader


def ngram_reader(file_name, word_dict, is_infer, gram_num):
    """
    create reader for N-Gram.

    :param file_name: file name.
    :param N: N-Gram's N.
    :param word_dict: vocab with content of '{word, id}',
        'word' is string type , 'id' is int type.
    :return: data reader.
    """
    assert gram_num >= 2

    def reader():
        ids = []
        UNK_ID = word_dict['<unk>']
        with open(file_name) as file:
            for line in file:
                words = line.strip().split()
                if len(words) < gram_num + 1: continue
                ids = [word_dict.get(w, UNK_ID) for w in words]
                for i in range(len(ids) - gram_num - 1):
                    if is_infer:
                        yield tuple(ids[i:i + gram_num])
                    else:
                        yield tuple(ids[i:i + gram_num + 1])

    return reader


if __name__ == "__main__":
    from utils import load_dict
    word_dict = load_dict("data/vocab_cn.txt")
    for idx, data in enumerate(
            # rnn_reader(
            #     file_name="/home/caoying/opt/rsync/data/raw_shuffle_1.txt",
            #     word_dict=word_dict)()):
            ngram_reader(
                file_name="/home/caoying/opt/rsync/data/raw_shuffle_1.txt",
                word_dict=word_dict,
                is_infer=True,
                gram_num=4)()):
        print data
