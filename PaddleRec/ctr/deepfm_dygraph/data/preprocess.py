from __future__ import division
import os
import numpy
from collections import Counter
import shutil
import pickle


def get_raw_data(intput_file, raw_data, ins_per_file):
    if not os.path.isdir(raw_data):
        os.mkdir(raw_data)

    fin = open(intput_file, 'r')
    fout = open(os.path.join(raw_data, 'part-0'), 'w')
    for line_idx, line in enumerate(fin):
        if line_idx % ins_per_file == 0 and line_idx != 0:
            fout.close()
            cur_part_idx = int(line_idx / ins_per_file)
            fout = open(
                os.path.join(raw_data, 'part-' + str(cur_part_idx)), 'w')
        fout.write(line)
    fout.close()
    fin.close()


def split_data(raw_data, aid_data, train_data, test_data):
    split_rate_ = 0.9
    dir_train_file_idx_ = os.path.join(aid_data, 'train_file_idx.txt')
    filelist_ = [
        os.path.join(raw_data, 'part-%d' % x)
        for x in range(len(os.listdir(raw_data)))
    ]

    if not os.path.exists(dir_train_file_idx_):
        train_file_idx = list(
            numpy.random.choice(
                len(filelist_), int(len(filelist_) * split_rate_), False))
        with open(dir_train_file_idx_, 'w') as fout:
            fout.write(str(train_file_idx))
    else:
        with open(dir_train_file_idx_, 'r') as fin:
            train_file_idx = eval(fin.read())

    for idx in range(len(filelist_)):
        if idx in train_file_idx:
            shutil.move(filelist_[idx], train_data)
        else:
            shutil.move(filelist_[idx], test_data)


def get_feat_dict(intput_file, aid_data, print_freq=100000, total_ins=45000000):
    freq_ = 10
    dir_feat_dict_ = os.path.join(aid_data, 'feat_dict_' + str(freq_) + '.pkl2')
    continuous_range_ = range(1, 14)
    categorical_range_ = range(14, 40)

    if not os.path.exists(dir_feat_dict_):
        # print('generate a feature dict')
        # Count the number of occurrences of discrete features
        feat_cnt = Counter()
        with open(intput_file, 'r') as fin:
            for line_idx, line in enumerate(fin):
                if line_idx % print_freq == 0:
                    print(r'generating feature dict {:.2f} %'.format((
                        line_idx / total_ins) * 100))
                features = line.rstrip('\n').split('\t')
                for idx in categorical_range_:
                    if features[idx] == '': continue
                    feat_cnt.update([features[idx]])

        # Only retain discrete features with high frequency
        dis_feat_set = set()
        for feat, ot in feat_cnt.items():
            if ot >= freq_:
                dis_feat_set.add(feat)

        # Create a dictionary for continuous and discrete features
        feat_dict = {}
        tc = 1
        # Continuous features
        for idx in continuous_range_:
            feat_dict[idx] = tc
            tc += 1
        for feat in dis_feat_set:
            feat_dict[feat] = tc
            tc += 1
        # Save dictionary
        with open(dir_feat_dict_, 'wb') as fout:
            pickle.dump(feat_dict, fout, protocol=2)
        print('args.num_feat ', len(feat_dict) + 1)


def preprocess(input_file,
               outdir,
               ins_per_file,
               total_ins=None,
               print_freq=None):
    train_data = os.path.join(outdir, "train_data")
    test_data = os.path.join(outdir, "test_data")
    aid_data = os.path.join(outdir, "aid_data")
    raw_data = os.path.join(outdir, "raw_data")
    if not os.path.isdir(train_data):
        os.mkdir(train_data)
    if not os.path.isdir(test_data):
        os.mkdir(test_data)
    if not os.path.isdir(aid_data):
        os.mkdir(aid_data)

    if print_freq is None:
        print_freq = 10 * ins_per_file

    get_raw_data(input_file, raw_data, ins_per_file)
    split_data(raw_data, aid_data, train_data, test_data)
    get_feat_dict(input_file, aid_data, print_freq, total_ins)

    print('Done!')


if __name__ == '__main__':
    preprocess('train.txt', './', 200000, 45000000)
