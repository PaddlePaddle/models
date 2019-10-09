#!/usr/bin/env python
# -*- coding: utf-8 -*-
################################################################################
# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
################################################################################

import math
import numpy as np
from collections import defaultdict

import paddle
import paddle.fluid as fluid


def data2lodtensor(data, place):
    lod = []
    while isinstance(data[0], list):
        lod.append(list(map(len, data)))
        data = [x for xs in data for x in xs]
    array = np.array(data, dtype="int64")
    if len(array.shape) == 1:
        array = array[:, None]
    tensor = fluid.LoDTensor()
    tensor.set(array, place)
    if len(lod) > 0:
        tensor.set_recursive_sequence_lengths(lod)
    return tensor


class DataLoader(object):
    def __init__(self,
                 data,
                 batch_size,
                 shuffle=False,
                 buf_size=4096,
                 use_gpu=False):
        def data_reader():
            return data

        if shuffle:
            self.reader = paddle.batch(
                paddle.reader.shuffle(
                    data_reader, buf_size=buf_size),
                batch_size=batch_size)
        else:
            self.reader = paddle.batch(data_reader, batch_size=batch_size)
        self.num_batches = math.ceil(len(data) / batch_size)
        self.place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        for examples in self.reader():
            batch_size = len(examples)
            batch = defaultdict(list)

            for ex in examples:
                for k, v in ex.items():
                    batch[k].append(v)

            batch = {k: data2lodtensor(v, self.place) for k, v in batch.items()}
            batch["size"] = batch_size
            yield batch
