# -*- coding: utf-8 -*-
#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import io
import sys

import numpy as np
import paddle
from paddle.io import Dataset

Py3 = sys.version_info[0] == 3

UNK_ID = 0


def _read_words(filename):
    data = []
    with io.open(filename, "r", encoding='utf-8') as f:
        if Py3:
            return f.read().replace("\n", "<eos>").split()
        else:
            return f.read().decode("utf-8").replace(u"\n", u"<eos>").split()


def read_all_line(filenam):
    data = []
    with io.open(filename, "r", encoding='utf-8') as f:
        for line in f.readlines():
            data.append(line.strip())


def _build_vocab(filename):
    pad_id = 0
    vocab_dict = {}
    ids = 0
    with io.open(filename, "r", encoding='utf-8') as f:
        for line in f.readlines():
            word = line.strip()
            vocab_dict[word] = ids
            if word == '</s>':
                pad_id = ids
            ids += 1

    print("vocab word num", ids)

    return vocab_dict, pad_id


def _para_file_to_ids(src_file, tar_file, src_vocab, tar_vocab):

    src_data = []
    with io.open(src_file, "r", encoding='utf-8') as f_src:
        for line in f_src.readlines():
            arra = line.strip().split()
            ids = [src_vocab[w] if w in src_vocab else UNK_ID for w in arra]
            ids = ids
            src_data.append(ids)

    tar_data = []
    with io.open(tar_file, "r", encoding='utf-8') as f_tar:
        for line in f_tar.readlines():
            arra = line.strip().split()
            ids = [tar_vocab[w] if w in tar_vocab else UNK_ID for w in arra]
            ids = [1] + ids + [2]
            tar_data.append(ids)

    return src_data, tar_data


def filter_len(src, tar, max_sequence_len=50):
    new_src = []
    new_tar = []

    for id1, id2 in zip(src, tar):
        if len(id1) > max_sequence_len:
            id1 = id1[:max_sequence_len]
        if len(id2) > max_sequence_len + 2:
            id2 = id2[:max_sequence_len + 2]

        new_src.append(id1)
        new_tar.append(id2)

    return new_src, new_tar


def raw_data(src_lang,
             tar_lang,
             vocab_prefix,
             train_prefix,
             eval_prefix,
             test_prefix,
             max_sequence_len=50):

    src_vocab_file = vocab_prefix + "." + src_lang
    tar_vocab_file = vocab_prefix + "." + tar_lang

    src_train_file = train_prefix + "." + src_lang
    tar_train_file = train_prefix + "." + tar_lang

    src_eval_file = eval_prefix + "." + src_lang
    tar_eval_file = eval_prefix + "." + tar_lang

    src_test_file = test_prefix + "." + src_lang
    tar_test_file = test_prefix + "." + tar_lang

    src_vocab, src_pad_id = _build_vocab(src_vocab_file)
    tar_vocab, tar_pad_id = _build_vocab(tar_vocab_file)

    train_src, train_tar = _para_file_to_ids( src_train_file, tar_train_file, \
                                              src_vocab, tar_vocab )
    train_src, train_tar = filter_len(
        train_src, train_tar, max_sequence_len=max_sequence_len)
    eval_src, eval_tar = _para_file_to_ids( src_eval_file, tar_eval_file, \
                                              src_vocab, tar_vocab )

    test_src, test_tar = _para_file_to_ids( src_test_file, tar_test_file, \
                                              src_vocab, tar_vocab )

    return (train_src, train_tar), (eval_src, eval_tar), (test_src, test_tar),\
            (src_vocab, tar_vocab), (src_pad_id, tar_pad_id)


def raw_mono_data(vocab_file, file_path):
    src_vocab = _build_vocab(vocab_file)
    test_src, test_tar = _para_file_to_ids( file_path, file_path, \
                                              src_vocab, src_vocab )

    return (test_src, test_tar)


class IWSLTDataset(Dataset):
    def __init__(self, raw_data):
        super(IWSLTDataset, self).__init__()
        self.raw_data = raw_data
        self.src_data, self.tar_data = self.sort_data(raw_data)
        self.num_samples = len(self.src_data)

    def sort_data(self, raw_data):
        src_data, trg_data = raw_data
        data_pair = []
        for src, trg in zip(src_data, trg_data):
            if len(src) > 0:
                data_pair.append([src, trg])
        sorted_data_pair = sorted(data_pair, key=lambda k: len(k[0]))

        src_data = [data_pair[0] for data_pair in sorted_data_pair]
        trg_data = [data_pair[1] for data_pair in sorted_data_pair]

        return src_data, trg_data

    def __getitem__(self, idx):
        src_ids, tar_ids = np.asarray(self.src_data[idx]), np.asarray(
            self.tar_data[idx])
        src_mask, tar_mask = len(src_ids), len(tar_ids)
        return src_ids, tar_ids, src_mask, tar_mask

    def __len__(self):
        return self.num_samples


class DataCollector():
    def __init__(self, pad_ids=(0, 0)):
        super(DataCollector, self).__init__()
        self.pad_ids = pad_ids

    def __call__(self, samples):
        batch_size = len(samples)

        src_ids_np = np.asarray([sample[0] for sample in samples])
        tar_ids_np = np.asarray([sample[1] for sample in samples])
        src_len = np.asarray([sample[2] for sample in samples])
        trg_len = np.asarray([sample[3] for sample in samples])

        max_source_len = max(max(src_len), 1)
        max_tar_len = max(max(trg_len), 1)

        src_ids_pad_np = self.pad(src_ids_np, max_source_len, source=True)
        tar_ids_pad_np = self.pad(tar_ids_np, max_tar_len)

        in_tar = tar_ids_pad_np[:, :-1]
        label_tar = tar_ids_pad_np[:, 1:]
        label_tar = label_tar.reshape(
            (label_tar.shape[0], label_tar.shape[1], 1))

        input_data_feed = src_ids_pad_np, in_tar, label_tar, src_len, trg_len
        return input_data_feed

    def pad(self, data, max_len, source=False):
        if source:
            pad_id = self.pad_ids[0]
        else:
            pad_id = self.pad_ids[1]
        bs = len(data)
        ids = np.ones((bs, max_len), dtype='int64') * pad_id

        for i, ele in enumerate(data):
            ids[i, :len(ele)] = ele

        return ids
