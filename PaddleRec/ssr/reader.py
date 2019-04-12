#Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random


class Dataset:
    def __init__(self):
        pass


class Vocab:
    def __init__(self):
        pass


class YoochooseVocab(Vocab):
    def __init__(self):
        self.vocab = {}
        self.word_array = []

    def load(self, filelist):
        idx = 0
        for f in filelist:
            with open(f, "r") as fin:
                for line in fin:
                    group = line.strip().split()
                    for item in group:
                        if item not in self.vocab:
                            self.vocab[item] = idx
                            self.word_array.append(idx)
                            idx += 1
                        else:
                            self.word_array.append(self.vocab[item])

    def get_vocab(self):
        return self.vocab

    def _get_word_array(self):
        return self.word_array


class YoochooseDataset(Dataset):
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size

    def sample_neg(self):
        return random.randint(0, self.vocab_size - 1)

    def sample_neg_from_seq(self, seq):
        return seq[random.randint(0, len(seq) - 1)]

    def _reader_creator(self, filelist, is_train):
        def reader():
            for f in filelist:
                with open(f, 'r') as fin:
                    line_idx = 0
                    for line in fin:
                        ids = line.strip().split()
                        if len(ids) <= 1:
                            continue
                        conv_ids = [i for i in ids]
                        boundary = len(ids) - 1
                        src = conv_ids[:boundary]
                        pos_tgt = [conv_ids[boundary]]
                        if is_train:
                            neg_tgt = [self.sample_neg()]
                            yield [src, pos_tgt, neg_tgt]
                        else:
                            yield [src, pos_tgt]

        return reader

    def train(self, file_list):
        return self._reader_creator(file_list, True)

    def test(self, file_list):
        return self._reader_creator(file_list, False)
