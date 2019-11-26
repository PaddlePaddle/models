import os
import numpy
from collections import Counter
import shutil
import pickle

SPLIT_RATIO = 0.9
INPUT_FILE = 'dist_data_demo.txt'
TRAIN_FILE = os.path.join('dist_train_data', 'tr')
TEST_FILE = os.path.join('dist_test_data', 'ev')


def split_data():
    all_lines = []
    for line in open(INPUT_FILE):
        all_lines.append(line)
    split_line_idx = int(len(all_lines) * SPLIT_RATIO)
    with open(TRAIN_FILE, 'w') as f:
        f.writelines(all_lines[:split_line_idx])
    with open(TEST_FILE, 'w') as f:
        f.writelines(all_lines[split_line_idx:])


def get_feat_dict():
    freq_ = 10
    dir_feat_dict_ = 'aid_data/feat_dict_' + str(freq_) + '.pkl2'
    continuous_range_ = range(1, 14)
    categorical_range_ = range(14, 40)

    if not os.path.exists(dir_feat_dict_):
        # print('generate a feature dict')
        # Count the number of occurrences of discrete features
        feat_cnt = Counter()
        with open(INPUT_FILE, 'r') as fin:
            for line_idx, line in enumerate(fin):
                features = line.rstrip('\n').split('\t')
                for idx in categorical_range_:
                    if features[idx] == '': continue
                    feat_cnt.update([features[idx]])

        # Only retain discrete features with high frequency
        # not filter low freq in small dataset
        freq_ = 0
        feat_set = set()
        for feat, ot in feat_cnt.items():
            if ot >= freq_:
                feat_set.add(feat)

        # Create a dictionary for continuous and discrete features
        feat_dict = {}
        tc = 1
        # Continuous features
        for idx in continuous_range_:
            feat_dict[idx] = tc
            tc += 1
        for feat in feat_set:
            feat_dict[feat] = tc
            tc += 1
        with open(dir_feat_dict_, 'wb') as fout:
            pickle.dump(feat_dict, fout, protocol=2)
        print('args.num_feat ', len(feat_dict) + 1)


if __name__ == '__main__':
    if not os.path.isdir('dist_train_data'):
        os.mkdir('dist_train_data')
    if not os.path.isdir('dist_test_data'):
        os.mkdir('dist_test_data')
    if not os.path.isdir('aid_data'):
        os.mkdir('aid_data')

    split_data()
    get_feat_dict()

    print('Done!')
