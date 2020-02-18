from __future__ import print_function, absolute_import, division

import os
import sys
from collections import Counter
import numpy as np
"""
preprocess Criteo train data, generate extra statistic files for model input.
"""
# input filename
FILENAME = 'dist_data_demo.txt'

# global vars
CAT_FEATURE_NUM = 'cat_feature_num.txt'
INT_FEATURE_MINMAX = 'int_feature_minmax.txt'
VOCAB_DIR = 'vocab'
TRAIN_DIR = 'dist_train_data'
TEST_DIR = 'dist_test_valid_data'
TRAIN_FILE = os.path.join(TRAIN_DIR, 'tr')
TEST_FILE = os.path.join(TEST_DIR, 'ev')
SPLIT_RATIO = 0.9
FREQ_THR = 10

INT_COLUMN_NAMES = ['I' + str(i) for i in range(1, 14)]
CAT_COLUMN_NAMES = ['C' + str(i) for i in range(1, 27)]


def check_statfiles():
    """
    check if statistic files of Criteo exists
    :return:
    """
    statsfiles = [CAT_FEATURE_NUM, INT_FEATURE_MINMAX] + [
        os.path.join(VOCAB_DIR, cat_fn + '.txt') for cat_fn in CAT_COLUMN_NAMES
    ]
    if all([os.path.exists(fn) for fn in statsfiles]):
        return True
    return False


def create_statfiles():
    """
    create statistic files of Criteo, including:
    min/max of interger features
    counts of categorical features
    vocabs of each categorical features
    :return:
    """
    int_minmax_list = [[sys.maxsize, -sys.maxsize]
                       for _ in range(13)]  # count integer feature min max
    cat_ct_list = [Counter() for _ in range(26)]  # count categorical features
    for idx, line in enumerate(open(FILENAME)):
        spls = line.rstrip('\n').split('\t')
        assert len(spls) == 40

        for i in range(13):
            if not spls[1 + i]: continue
            int_val = int(spls[1 + i])
            int_minmax_list[i][0] = min(int_minmax_list[i][0], int_val)
            int_minmax_list[i][1] = max(int_minmax_list[i][1], int_val)

        for i in range(26):
            cat_ct_list[i].update([spls[14 + i]])

    # save min max of integer features
    with open(INT_FEATURE_MINMAX, 'w') as f:
        for name, minmax in zip(INT_COLUMN_NAMES, int_minmax_list):
            print("{} {} {}".format(name, minmax[0], minmax[1]), file=f)

    # remove '' from all cat_set[i] and filter low freq categorical value
    cat_set_list = [set() for i in range(len(cat_ct_list))]
    for i, ct in enumerate(cat_ct_list):
        if '' in ct: del ct['']
        for key in list(ct.keys()):
            if ct[key] >= FREQ_THR:
                cat_set_list[i].add(key)

    del cat_ct_list

    # create vocab dir
    if not os.path.exists(VOCAB_DIR):
        os.makedirs(VOCAB_DIR)

    # write vocab file of categorical features
    with open(CAT_FEATURE_NUM, 'w') as cat_feat_count_file:
        for name, s in zip(CAT_COLUMN_NAMES, cat_set_list):
            print('{} {}'.format(name, len(s)), file=cat_feat_count_file)

            vocabfile = os.path.join(VOCAB_DIR, name + '.txt')

            with open(vocabfile, 'w') as f:
                for vocab_val in s:
                    print(vocab_val, file=f)


def split_data():
    """
    split train.txt into train and test_valid files.
    :return:
    """
    if not os.path.exists(TRAIN_DIR):
        os.makedirs(TRAIN_DIR)
    if not os.path.exists(TEST_DIR):
        os.makedirs(TEST_DIR)

    all_lines = []
    for line in open(FILENAME):
        all_lines.append(line)
    split_line_idx = int(len(all_lines) * SPLIT_RATIO)
    with open(TRAIN_FILE, 'w') as f:
        f.writelines(all_lines[:split_line_idx])
    with open(TEST_FILE, 'w') as f:
        f.writelines(all_lines[split_line_idx:])


if __name__ == '__main__':
    if not check_statfiles():
        print('create statstic files of Criteo...')
        create_statfiles()
    print('split train.txt...')
    split_data()
    print('done')
