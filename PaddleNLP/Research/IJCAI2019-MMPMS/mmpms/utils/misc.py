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

import numpy as np
import paddle.fluid as fluid


def tensor2list(T):
    lod = T.lod()
    array = np.array(T)
    if array.shape[-1] == 1:
        array = array.squeeze(-1)
    array = array.tolist()
    for lod_i in lod[::-1]:
        array = [array[start:end] for start, end in zip(lod_i, lod_i[1:])]
    return array


def sequence_last(T, place):
    lod = T.lod()[-1]
    recursive_seq_lens = T.recursive_sequence_lengths()
    array = np.array(T)
    last_ids = np.array(lod[1:]) - 1
    data = array[last_ids]
    return fluid.create_lod_tensor(data, recursive_seq_lens[:-1], place)


def sequence_but(T, place, position="first"):
    assert position in ["first", "last"]
    lod = T.lod()[-1][1:-1]
    recursive_seq_lens = T.recursive_sequence_lengths()
    array = np.array(T)
    if position == "first":
        data = np.concatenate([a[1:] for a in np.split(array, lod)])
    else:
        data = np.concatenate([a[:-1] for a in np.split(array, lod)])
    recursive_seq_lens[-1] = [l - 1 for l in recursive_seq_lens[-1]]
    return fluid.create_lod_tensor(data, recursive_seq_lens, place)
