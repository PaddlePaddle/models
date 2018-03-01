import collections
import os

MIN_LEN = 3
MAX_LEN = 100


def rnn_reader(file_name, word_dict):
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
                words = line.strip().lower().split()
                if len(words) < MIN_LEN or len(words) > MAX_LEN:
                    continue
                ids = [word_dict.get(w, UNK_ID)
                       for w in words] + [word_dict['<e>']]
                yield ids[:-1], ids[1:]

    return reader
