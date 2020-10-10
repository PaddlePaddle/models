import sys
import paddle.fluid.incubate.data_generator as dg
try:
    import cPickle as pickle
except ImportError:
    import pickle
from collections import Counter
import os


class CriteoDataset(dg.MultiSlotDataGenerator):
    def setup(self, feat_dict_name):
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
        self.feat_dict_ = pickle.load(open(feat_dict_name, 'rb'))

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

    def test(self, filelist):
        def local_iter():
            for fname in filelist:
                with open(fname.strip(), 'r') as fin:
                    for line in fin:
                        feat_idx, feat_value, label = self._process_line(line)
                        yield [feat_idx, feat_value, label]

        return local_iter

    def generate_sample(self, line):
        def data_iter():
            feat_idx, feat_value, label = self._process_line(line)
            yield [('feat_idx', feat_idx), ('feat_value', feat_value), ('label',
                                                                        label)]

        return data_iter


if __name__ == '__main__':
    import paddle
    paddle.enable_static()
    criteo_dataset = CriteoDataset()
    if len(sys.argv) <= 1:
        sys.stderr.write("feat_dict needed for criteo reader.")
        exit(1)
    criteo_dataset.setup(sys.argv[1])
    criteo_dataset.run_from_stdin()
