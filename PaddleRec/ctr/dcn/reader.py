"""
dataset and reader
"""
import math
import sys
import paddle.fluid.incubate.data_generator as dg
import pickle
from collections import Counter
import os


class CriteoDataset(dg.MultiSlotDataGenerator):
    def setup(self, vocab_dir):
        self.cont_min_ = [0, -3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.cont_max_ = [
            5775, 257675, 65535, 969, 23159456, 431037, 56311, 6047, 29019, 11,
            231, 4008, 7393
        ]
        self.cont_diff_ = [
            self.cont_max_[i] - self.cont_min_[i]
            for i in range(len(self.cont_min_))
        ]
        self.cont_idx_ = list(range(1, 14))
        self.cat_idx_ = list(range(14, 40))

        dense_feat_names = ['I' + str(i) for i in range(1, 14)]
        sparse_feat_names = ['C' + str(i) for i in range(1, 27)]
        target = ['label']

        self.label_feat_names = target + dense_feat_names + sparse_feat_names

        self.cat_feat_idx_dict_list = [{} for _ in range(26)]
        for i in range(26):
            lookup_idx = 1  # remain 0 for default value
            for line in open(
                    os.path.join(vocab_dir, 'C' + str(i + 1) + '.txt')):
                self.cat_feat_idx_dict_list[i][line.strip()] = lookup_idx
                lookup_idx += 1

    def _process_line(self, line):
        features = line.rstrip('\n').split('\t')
        label_feat_list = [[] for _ in range(40)]
        for idx in self.cont_idx_:
            if features[idx] == '':
                label_feat_list[idx].append(0)
            else:
                # 0-1 minmax norm
                # label_feat_list[idx].append((float(features[idx]) - self.cont_min_[idx - 1]) /
                #                             self.cont_diff_[idx - 1])
                # log transform
                label_feat_list[idx].append(
                    math.log(4 + float(features[idx]))
                    if idx == 2 else math.log(1 + float(features[idx])))
        for idx in self.cat_idx_:
            if features[idx] == '' or features[
                    idx] not in self.cat_feat_idx_dict_list[idx - 14]:
                label_feat_list[idx].append(0)
            else:
                label_feat_list[idx].append(self.cat_feat_idx_dict_list[
                    idx - 14][features[idx]])
        label_feat_list[0].append(int(features[0]))
        return label_feat_list

    def test_reader(self, filelist, batch, buf_size):
        print(filelist)

        def local_iter():
            for fname in filelist:
                with open(fname.strip(), 'r') as fin:
                    for line in fin:
                        label_feat_list = self._process_line(line)
                        yield label_feat_list

        import paddle
        batch_iter = paddle.batch(
            paddle.reader.buffered(
                local_iter, size=buf_size), batch_size=batch)
        return batch_iter

    def generate_sample(self, line):
        def data_iter():
            label_feat_list = self._process_line(line)
            yield list(zip(self.label_feat_names, label_feat_list))

        return data_iter


if __name__ == '__main__':
    criteo_dataset = CriteoDataset()
    if len(sys.argv) <= 1:
        sys.stderr.write("feat_dict needed for criteo reader.")
        exit(1)
    criteo_dataset.setup(sys.argv[1])
    criteo_dataset.run_from_stdin()
