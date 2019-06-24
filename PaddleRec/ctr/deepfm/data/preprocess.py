import os
import pickle
import numpy
from collections import Counter


def get_raw_data():
    fin = open('train.txt', 'r')
    fout = open('raw_data/part-0', 'w')
    for line_idx, line in enumerate(fin):
        if line_idx % 200000 == 0 and line_idx != 0:
            fout.close()
            cur_part_idx = int(line_idx / 200000)
            fout = open('raw_data/part-' + str(cur_part_idx), 'w')
        fout.write(line)
    fout.close()
    fin.close()


def split_data():
    split_rate_ = 0.9
    dir_train_file_idx_ = 'aid_data/train_file_idx.pkl2'
    filelist_ = [
        'raw_data/part-%d' % x for x in range(len(os.listdir('raw_data')))
    ]

    if not os.path.exists(dir_train_file_idx_):
        train_file_idx = list(
            numpy.random.choice(
                len(filelist_), int(len(filelist_) * split_rate_), False))
        with open(dir_train_file_idx_, 'wb') as fout:
            pickle.dump(train_file_idx, fout)


def get_feat_dict():
    freq_ = 10
    dir_feat_dict_ = 'aid_data/feat_dict_' + str(freq_) + '.pkl2'
    filelist_ = [
        'raw_data/part-%d' % x for x in range(len(os.listdir('raw_data')))
    ]
    continuous_range_ = range(1, 14)
    categorical_range_ = range(14, 40)

    if not os.path.exists(dir_feat_dict_):
        # print('generate a feature dict')
        # Count the number of occurrences of discrete features
        feat_cnt = Counter()
        for fname in filelist_:
            with open(fname.strip(), 'r') as fin:
                for line_idx, line in enumerate(fin):
                    if line_idx == 0: print('generating feature dict')
                    features = line.lstrip('\n').split('\t')
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
        # Discrete features
        cnt_feat_set = set()
        for fname in filelist_:
            with open(fname.strip(), 'r') as fin:
                for line_idx, line in enumerate(fin):
                    features = line.rstrip('\n').split('\t')
                    for idx in categorical_range_:
                        if features[idx] == '' or features[
                                idx] not in dis_feat_set:
                            continue
                        if features[idx] not in cnt_feat_set:
                            cnt_feat_set.add(features[idx])
                            feat_dict[features[idx]] = tc
                            tc += 1

        # Save dictionary
        with open(dir_feat_dict_, 'wb') as fout:
            pickle.dump(feat_dict, fout)
        print('args.num_feat ', len(feat_dict) + 1)


if __name__ == '__main__':
    if not os.path.isdir('raw_data'):
        os.mkdir('raw_data')
    if not os.path.isdir('aid_data'):
        os.mkdir('aid_data')

    get_raw_data()
    split_data()
    get_feat_dict()

    print('Done!')
