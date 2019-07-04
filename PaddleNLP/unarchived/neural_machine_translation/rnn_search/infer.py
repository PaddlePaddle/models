#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
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

import numpy as np
import time
import os
import random

import math

import paddle
import paddle.fluid as fluid
import paddle.fluid.framework as framework
from paddle.fluid.executor import Executor

import reader

import sys
if sys.version[0] == '2':
    reload(sys)
    sys.setdefaultencoding("utf-8")
import os

from args import *
#from . import lm_model
import logging
import pickle

from attention_model import AttentionModel

from base_model import BaseModel

SEED = 123


def train():
    args = parse_args()

    num_layers = args.num_layers
    src_vocab_size = args.src_vocab_size
    tar_vocab_size = args.tar_vocab_size
    batch_size = args.batch_size
    dropout = args.dropout
    init_scale = args.init_scale
    max_grad_norm = args.max_grad_norm
    hidden_size = args.hidden_size
    # inference process

    print("src", src_vocab_size)

    # dropout type using upscale_in_train, dropout can be remove in inferecen
    # So we can set dropout to 0
    if args.attention:
        model = AttentionModel(
            hidden_size,
            src_vocab_size,
            tar_vocab_size,
            batch_size,
            num_layers=num_layers,
            init_scale=init_scale,
            dropout=0.0)
    else:
        model = BaseModel(
            hidden_size,
            src_vocab_size,
            tar_vocab_size,
            batch_size,
            num_layers=num_layers,
            init_scale=init_scale,
            dropout=0.0)

    beam_size = args.beam_size
    trans_res = model.build_graph(mode='beam_search', beam_size=beam_size)
    # clone from default main program and use it as the validation program
    main_program = fluid.default_main_program()

    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = Executor(place)
    exe.run(framework.default_startup_program())

    source_vocab_file = args.vocab_prefix + "." + args.src_lang
    infer_file = args.infer_file

    infer_data = reader.raw_mono_data(source_vocab_file, infer_file)

    def prepare_input(batch, epoch_id=0, with_lr=True):
        src_ids, src_mask, tar_ids, tar_mask = batch
        res = {}
        src_ids = src_ids.reshape((src_ids.shape[0], src_ids.shape[1], 1))
        in_tar = tar_ids[:, :-1]
        label_tar = tar_ids[:, 1:]

        in_tar = in_tar.reshape((in_tar.shape[0], in_tar.shape[1], 1))
        in_tar = np.zeros_like(in_tar, dtype='int64')
        label_tar = label_tar.reshape(
            (label_tar.shape[0], label_tar.shape[1], 1))
        label_tar = np.zeros_like(label_tar, dtype='int64')

        res['src'] = src_ids
        res['tar'] = in_tar
        res['label'] = label_tar
        res['src_sequence_length'] = src_mask
        res['tar_sequence_length'] = tar_mask

        return res, np.sum(tar_mask)

    dir_name = args.reload_model
    print("dir name", dir_name)
    fluid.io.load_params(exe, dir_name)

    train_data_iter = reader.get_data_iter(infer_data, 1, mode='eval')

    tar_id2vocab = []
    tar_vocab_file = args.vocab_prefix + "." + args.tar_lang
    with open(tar_vocab_file, "r") as f:
        for line in f.readlines():
            tar_id2vocab.append(line.strip())

    infer_output_file = args.infer_output_file

    out_file = open(infer_output_file, 'w')

    for batch_id, batch in enumerate(train_data_iter):
        input_data_feed, word_num = prepare_input(batch, epoch_id=0)

        fetch_outs = exe.run(feed=input_data_feed,
                             fetch_list=[trans_res.name],
                             use_program_cache=False)

        res = [tar_id2vocab[e] for e in fetch_outs[0].reshape(-1)]

        res = res[1:]

        new_res = []
        for ele in res:
            if ele == "</s>":
                break
            new_res.append(ele)

        out_file.write(' '.join(new_res))
        out_file.write('\n')

    out_file.close()


if __name__ == '__main__':
    train()
