#coding=utf-8

import random


def load_dict(dict_file):
    word_dict = dict()
    with open(dict_file, 'r') as f:
        for i, line in enumerate(f):
            w = line.strip().split()[0]
            word_dict[w] = i
    return word_dict


def get_reverse_dict(dictionary):
    reverse_dict = {dictionary[k]: k for k in dictionary.keys()}
    return reverse_dict


def load_data(data_file, src_dict, trg_dict):
    UNK_IDX = src_dict['UNK']
    with open(data_file, 'r') as f:
        for line in f:
            line_split = line.strip().split('\t')
            if len(line_split) < 2:
                continue
            src, trg = line_split
            src_words = src.strip().split()
            trg_words = trg.strip().split()
            src_seq = [src_dict.get(w, UNK_IDX) for w in src_words]
            trg_seq = [trg_dict.get(w, UNK_IDX) for w in trg_words]
            yield src_seq, trg_seq


def data_reader(data_file, src_dict, trg_dict, pos_size, padding_num):
    def reader():
        UNK_IDX = src_dict['UNK']
        word_padding = trg_dict.__len__()
        pos_padding = pos_size

        def _get_pos(pos_list, pos_size, pos_padding):
            return [pos if pos < pos_size else pos_padding for pos in pos_list]

        with open(data_file, 'r') as f:
            for line in f:
                line_split = line.strip().split('\t')
                if len(line_split) != 2:
                    continue
                src, trg = line_split
                src = src.strip().split()
                src_word = [src_dict.get(w, UNK_IDX) for w in src]
                src_word_pos = range(len(src_word))
                src_word_pos = _get_pos(src_word_pos, pos_size, pos_padding)

                trg = trg.strip().split()
                trg_word = [trg_dict['<s>']
                            ] + [trg_dict.get(w, UNK_IDX) for w in trg]
                trg_word_pos = range(len(trg_word))
                trg_word_pos = _get_pos(trg_word_pos, pos_size, pos_padding)

                trg_next_word = trg_word[1:] + [trg_dict['<e>']]
                trg_word = [word_padding] * padding_num + trg_word
                trg_word_pos = [pos_padding] * padding_num + trg_word_pos
                trg_next_word = trg_next_word + [trg_dict['<e>']] * padding_num
                yield src_word, src_word_pos, trg_word, trg_word_pos, trg_next_word

    return reader
