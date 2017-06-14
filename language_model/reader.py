# coding=utf-8
import collections
import os


def rnn_reader(file_name, min_sentence_length, max_sentence_length,
               word_id_dict):
    """
    create reader for RNN, each line is a sample.

    :param file_name: file name.
    :param min_sentence_length: sentence's min length.
    :param max_sentence_length: sentence's max length.
    :param word_id_dict: vocab with content of '{word, id}', 'word' is string type , 'id' is int type.
    :return: data reader.
    """

    def reader():
        UNK = word_id_dict['<UNK>']
        with open(file_name) as file:
            for line in file:
                words = line.decode('utf-8', 'ignore').strip().split()
                if len(words) < min_sentence_length or len(
                        words) > max_sentence_length:
                    continue
                ids = [word_id_dict.get(w, UNK) for w in words]
                ids.append(word_id_dict['<EOS>'])
                target = ids[1:]
                target.append(word_id_dict['<EOS>'])
                yield ids[:], target[:]

    return reader


def ngram_reader(file_name, N, word_id_dict):
    """
    create reader for N-Gram.

    :param file_name: file name.
    :param N: N-Gram's N.
    :param word_id_dict: vocab with content of '{word, id}', 'word' is string type , 'id' is int type.
    :return: data reader.
    """
    assert N >= 2

    def reader():
        ids = []
        UNK_ID = word_id_dict['<UNK>']
        cache_size = 10000000
        with open(file_name) as file:
            for line in file:
                words = line.decode('utf-8', 'ignore').strip().split()
                ids += [word_id_dict.get(w, UNK_ID) for w in words]
                ids_len = len(ids)
                if ids_len > cache_size:  # output
                    for i in range(ids_len - N - 1):
                        yield tuple(ids[i:i + N])
                    ids = []
        ids_len = len(ids)
        for i in range(ids_len - N - 1):
            yield tuple(ids[i:i + N])

    return reader
