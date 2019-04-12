#  Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

import os
import random
import numpy as np
import paddle
import pickle

def pad_batch_data(input, max_len):
    res = np.array([x + [0] * (max_len - len(x)) for x in input])
    res = res.astype("int64").reshape([-1, max_len, 1])
    return res


def make_data(b):
    max_len = max(len(x[0]) for x in b)
    item = pad_batch_data([x[0] for x in b], max_len)
    cat = pad_batch_data([x[1] for x in b], max_len)
    len_array = [len(x[0]) for x in b]
    mask = np.array(
        [[0] * x + [-1e9] * (max_len - x) for x in len_array]).reshape(
            [-1, max_len, 1])
    target_item_seq = np.array(
        [[x[2]] * max_len for x in b]).astype("int64").reshape(
            [-1, max_len, 1])
    target_cat_seq = np.array(
        [[x[3]] * max_len for x in b]).astype("int64").reshape(
            [-1, max_len, 1])
    res = []
    for i in range(len(b)):
        res.append([
            item[i], cat[i], b[i][2], b[i][3], b[i][4], mask[i],
            target_item_seq[i], target_cat_seq[i]
        ])
    return res


def batch_reader(reader, batch_size, group_size):
    def batch_reader():
        bg = []
        for line in reader:
            bg.append(line)
            if len(bg) == group_size:
                sortb = sorted(bg, key=lambda x: len(x[0]), reverse=False)
                bg = []
                for i in range(0, group_size, batch_size):
                    b = sortb[i:i + batch_size]
                    yield make_data(b)
        len_bg = len(bg)
        if len_bg != 0:
            sortb = sorted(bg, key=lambda x: len(x[0]), reverse=False)
            bg = []
            remain = len_bg % batch_size
            for i in range(0, len_bg - remain, batch_size):
                b = sortb[i:i + batch_size]
                yield make_data(b)

    return batch_reader


def base_read(file_dir):
    res = []
    max_len = 0
    with open(file_dir, "r") as fin:
        for line in fin:
            line = line.strip().split(';')
            hist = line[0].split()
            cate = line[1].split()
            max_len = max(max_len, len(hist))
            res.append([hist, cate, line[2], line[3], float(line[4])])
    return res, max_len


def prepare_reader(data_path, bs):
    data_set, max_len = base_read(data_path)
    random.shuffle(data_set)
    return batch_reader(data_set, bs, bs * 20), max_len


def config_read(config_path):
    with open(config_path, "r") as fin:
        user_count = int(fin.readline().strip())
        item_count = int(fin.readline().strip())
        cat_count = int(fin.readline().strip())
    return user_count, item_count, cat_count
