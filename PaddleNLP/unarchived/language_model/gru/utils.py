#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
import sys
import time
import numpy as np

import paddle.fluid as fluid
import paddle


def to_lodtensor(data, place):
    """ convert to LODtensor """
    seq_lens = [len(seq) for seq in data]
    cur_len = 0
    lod = [cur_len]
    for l in seq_lens:
        cur_len += l
        lod.append(cur_len)
    flattened_data = np.concatenate(data, axis=0).astype("int64")
    flattened_data = flattened_data.reshape([len(flattened_data), 1])
    res = fluid.LoDTensor()
    res.set(flattened_data, place)
    res.set_lod([lod])
    return res


def prepare_data(batch_size,
                 buffer_size=1000,
                 word_freq_threshold=0,
                 enable_ce=False):
    """ prepare the English Pann Treebank (PTB) data """
    vocab = paddle.dataset.imikolov.build_dict(word_freq_threshold)
    if enable_ce:
        train_reader = paddle.batch(
            paddle.dataset.imikolov.train(
                vocab,
                buffer_size,
                data_type=paddle.dataset.imikolov.DataType.SEQ),
            batch_size)
    else:
        train_reader = paddle.batch(
            paddle.reader.shuffle(
                paddle.dataset.imikolov.train(
                    vocab,
                    buffer_size,
                    data_type=paddle.dataset.imikolov.DataType.SEQ),
                buf_size=buffer_size),
            batch_size)
    test_reader = paddle.batch(
        paddle.dataset.imikolov.test(
            vocab, buffer_size, data_type=paddle.dataset.imikolov.DataType.SEQ),
        batch_size)
    return vocab, train_reader, test_reader
