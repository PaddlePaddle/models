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
from base_model import BaseModel
from attention_model import AttentionModel
import logging
import pickle

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

    if args.enable_ce:
        fluid.default_main_program().random_seed = 102
        framework.default_startup_program().random_seed = 102

    # Training process

    if args.attention:
        model = AttentionModel(
            hidden_size,
            src_vocab_size,
            tar_vocab_size,
            batch_size,
            num_layers=num_layers,
            init_scale=init_scale,
            dropout=dropout)
    else:
        model = BaseModel(
            hidden_size,
            src_vocab_size,
            tar_vocab_size,
            batch_size,
            num_layers=num_layers,
            init_scale=init_scale,
            dropout=dropout)

    loss = model.build_graph()
    # clone from default main program and use it as the validation program
    main_program = fluid.default_main_program()
    inference_program = fluid.default_main_program().clone(for_test=True)

    fluid.clip.set_gradient_clip(clip=fluid.clip.GradientClipByGlobalNorm(
        clip_norm=max_grad_norm))

    lr = args.learning_rate
    opt_type = args.optimizer
    if opt_type == "sgd":
        optimizer = fluid.optimizer.SGD(lr)
    elif opt_type == "adam":
        optimizer = fluid.optimizer.Adam(lr)
    else:
        print("only support [sgd|adam]")
        raise Exception("opt type not support")

    optimizer.minimize(loss)

    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = Executor(place)
    exe.run(framework.default_startup_program())

    train_data_prefix = args.train_data_prefix
    eval_data_prefix = args.eval_data_prefix
    test_data_prefix = args.test_data_prefix
    vocab_prefix = args.vocab_prefix
    src_lang = args.src_lang
    tar_lang = args.tar_lang
    print("begin to load data")
    raw_data = reader.raw_data(src_lang, tar_lang, vocab_prefix,
                               train_data_prefix, eval_data_prefix,
                               test_data_prefix, args.max_len)
    print("finished load data")
    train_data, valid_data, test_data, _ = raw_data

    def prepare_input(batch, epoch_id=0, with_lr=True):
        src_ids, src_mask, tar_ids, tar_mask = batch
        res = {}
        src_ids = src_ids.reshape((src_ids.shape[0], src_ids.shape[1], 1))
        in_tar = tar_ids[:, :-1]
        label_tar = tar_ids[:, 1:]

        in_tar = in_tar.reshape((in_tar.shape[0], in_tar.shape[1], 1))
        label_tar = label_tar.reshape(
            (label_tar.shape[0], label_tar.shape[1], 1))

        res['src'] = src_ids
        res['tar'] = in_tar
        res['label'] = label_tar
        res['src_sequence_length'] = src_mask
        res['tar_sequence_length'] = tar_mask

        return res, np.sum(tar_mask)

    # get train epoch size
    def eval(data, epoch_id=0):
        eval_data_iter = reader.get_data_iter(data, batch_size, mode='eval')
        total_loss = 0.0
        word_count = 0.0
        for batch_id, batch in enumerate(eval_data_iter):
            input_data_feed, word_num = prepare_input(
                batch, epoch_id, with_lr=False)
            fetch_outs = exe.run(inference_program,
                                 feed=input_data_feed,
                                 fetch_list=[loss.name],
                                 use_program_cache=False)

            cost_train = np.array(fetch_outs[0])

            total_loss += cost_train * batch_size
            word_count += word_num

        ppl = np.exp(total_loss / word_count)

        return ppl

    ce_time = []
    ce_ppl = []
    max_epoch = args.max_epoch
    for epoch_id in range(max_epoch):
        start_time = time.time()
        print("epoch id", epoch_id)
        if args.enable_ce:
            train_data_iter = reader.get_data_iter(train_data, batch_size, enable_ce=True)
        else:
            train_data_iter = reader.get_data_iter(train_data, batch_size)
            

        total_loss = 0
        word_count = 0.0
        for batch_id, batch in enumerate(train_data_iter):

            input_data_feed, word_num = prepare_input(batch, epoch_id=epoch_id)
            fetch_outs = exe.run(feed=input_data_feed,
                                 fetch_list=[loss.name],
                                 use_program_cache=True)

            cost_train = np.array(fetch_outs[0])

            total_loss += cost_train * batch_size
            word_count += word_num

            if batch_id > 0 and batch_id % 100 == 0:
                print("ppl", batch_id, np.exp(total_loss / word_count))
                ce_ppl.append(np.exp(total_loss / word_count))
                total_loss = 0.0
                word_count = 0.0
        end_time = time.time()
        time_gap = end_time - start_time
        ce_time.append(time_gap)

        dir_name = args.model_path + "/epoch_" + str(epoch_id)
        print("begin to save", dir_name)
        fluid.io.save_params(exe, dir_name)
        print("save finished")
        dev_ppl = eval(valid_data)
        print("dev ppl", dev_ppl)
        test_ppl = eval(test_data)
        print("test ppl", test_ppl)

    if args.enable_ce:
        card_num = get_cards()
        _ppl = 0
        _time = 0
        try:
            _time = ce_time[-1]
            _ppl = ce_ppl[-1]
        except:
            print("ce info error")
        print("kpis\ttrain_duration_card%s\t%s" %
                (card_num, _time))
        print("kpis\ttrain_ppl_card%s\t%f" %
            (card_num, _ppl))


def get_cards():
    num = 0
    cards = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if cards != '':
        num = len(cards.split(","))
    return num


if __name__ == '__main__':
    train()
