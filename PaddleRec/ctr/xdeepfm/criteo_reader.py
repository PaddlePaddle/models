import sys
import paddle.fluid.incubate.data_generator as dg
import pickle
from collections import Counter
import os


class CriteoDataset(dg.MultiSlotDataGenerator):
    def _process_line(self, line):
        features = line.strip('\n').split('\t')
        feat_idx = []
        feat_value = []
        for idx in range(1, 40):
            feat_idx.append(int(features[idx]))
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
            feature_name = ['feat_idx', 'feat_value', 'label']
            yield [('feat_idx', feat_idx), ('feat_value', feat_value), ('label',
                                                                        label)]

        return data_iter


if __name__ == '__main__':
    import paddle
    paddle.enable_static()
    criteo_dataset = CriteoDataset()
    criteo_dataset.run_from_stdin()
