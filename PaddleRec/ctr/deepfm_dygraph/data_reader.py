from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
import random

import paddle


class DataGenerator(object):
    def __init__(self, feat_dict_path):
        # min-max of continuous features in Criteo dataset
        self.cont_min_ = [0, -3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.cont_max_ = [
            5775, 257675, 65535, 969, 23159456, 431037, 56311, 6047, 29019, 46,
            231, 4008, 7393
        ]
        self.cont_diff_ = [
            self.cont_max_[i] - self.cont_min_[i]
            for i in range(len(self.cont_min_))
        ]
        self.continuous_range_ = range(1, 14)
        self.categorical_range_ = range(14, 40)
        self.feat_dict_ = pickle.load(open(feat_dict_path, 'rb'))

    def _process_line(self, line):
        features = line.rstrip('\n').split('\t')
        feat_idx = []
        feat_value = []
        for idx in self.continuous_range_:
            if features[idx] == '':
                feat_idx.append(0)
                feat_value.append(0.0)
            else:
                feat_idx.append(self.feat_dict_[idx])
                feat_value.append(
                    (float(features[idx]) - self.cont_min_[idx - 1]) /
                    self.cont_diff_[idx - 1])
        for idx in self.categorical_range_:
            if features[idx] == '' or features[idx] not in self.feat_dict_:
                feat_idx.append(0)
                feat_value.append(0.0)
            else:
                feat_idx.append(self.feat_dict_[features[idx]])
                feat_value.append(1.0)
        label = [int(features[0])]
        return feat_idx, feat_value, label

    def train_reader(self, file_list, batch_size, cycle, shuffle=True):
        def _reader():
            if shuffle:
                random.shuffle(file_list)
            while True:
                for fn in file_list:
                    for line in open(fn, 'r'):
                        yield self._process_line(line)
                if not cycle:
                    break

        return paddle.batch(_reader, batch_size=batch_size)


def data_reader(batch_size,
                file_list,
                feat_dict_path,
                cycle=False,
                shuffle=False,
                data_type="train"):
    generator = DataGenerator(feat_dict_path)

    if data_type != "train" and data_type != "test":
        print("data type only support train | test")
        raise Exception("data type only support train | test")
    return generator.train_reader(file_list, batch_size, cycle, shuffle=shuffle)
