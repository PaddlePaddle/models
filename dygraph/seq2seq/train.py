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

import numpy as np
import time
import os
import logging
import random
import math
import contextlib

import paddle
from paddle.io import DataLoader, DistributedBatchSampler
import reader
from reader import IWSLTDataset, DataCollector

import sys
if sys.version[0] == '2':
    reload(sys)
    sys.setdefaultencoding("utf-8")

from args import *
from attention_model import AttentionModel
import logging
import pickle

SEED = 102
paddle.manual_seed(SEED)

args = parse_args()
paddle.set_default_dtype(args.dtype)
if args.enable_ce:
    np.random.seed(102)
    random.seed(102)


def create_model(args, pad_ids):
    model = AttentionModel(
        args.hidden_size,
        args.src_vocab_size,
        args.tar_vocab_size,
        num_layers=args.num_layers,
        init_scale=args.init_scale,
        pad_ids=pad_ids,
        dropout=args.dropout)
    return model


def create_data_loader(args, place, shuffle=True):
    print("begin to load data")
    raw_data = reader.raw_data(args.src_lang, args.tar_lang, args.vocab_prefix,
                               args.train_data_prefix, args.eval_data_prefix,
                               args.test_data_prefix, args.max_len)

    train_data, val_data, test_data, _, pad_ids = raw_data
    batch_fn = DataCollector(pad_ids)

    def _create_data_loader(data, batch_fn):
        dataset = IWSLTDataset(data)
        loader = DataLoader(
            dataset,
            places=place,
            return_list=True,
            batch_size=args.batch_size,
            shuffle=shuffle,
            collate_fn=batch_fn,
            drop_last=False)
        return loader

    train_loader = _create_data_loader(train_data, batch_fn)
    val_loader = _create_data_loader(val_data, batch_fn)
    test_loader = _create_data_loader(test_data, batch_fn)
    return train_loader, val_loader, test_loader, pad_ids


def main():
    print(args)
    place = paddle.set_device('gpu') if args.use_gpu else paddle.set_device(
        "cpu")
    paddle.disable_static()
    if args.enable_ce:
        np.random.seed(102)
        random.seed(102)

    train_loader, val_loader, test_loader, pad_ids = create_data_loader(
        args, place, shuffle=False)
    model = create_model(args, pad_ids)

    gloabl_norm_clip = paddle.nn.GradientClipByGlobalNorm(args.max_grad_norm)
    lr = args.learning_rate
    opt_type = args.optimizer

    if opt_type == "sgd":
        optimizer = paddle.optimizer.SGD(lr,
                                         parameters=model.parameters(),
                                         grad_clip=gloabl_norm_clip)
    elif opt_type == "adam":
        optimizer = paddle.optimizer.Adam(
            lr, parameters=model.parameters(), grad_clip=gloabl_norm_clip)
    else:
        print("only support [sgd|adam]")
        raise Exception("opt type not support")

    ce_time = []
    ce_ppl = []
    max_epoch = args.max_epoch
    for epoch_id in range(max_epoch):
        epoch_start = time.time()
        model.train()
        total_loss = 0
        word_count = 0.0
        batch_times, epoch_times, reader_times = [], [], []
        batch_start = time.time()
        for batch_id, input_data_feed in enumerate(train_loader):
            batch_reader_end = time.time()
            loss = model(input_data_feed)
            # print(loss.numpy()[0])
            optimizer.clear_grad()
            loss.backward()
            optimizer.step()

            train_batch_cost = time.time() - batch_start
            batch_times.append(train_batch_cost)
            epoch_times.append(train_batch_cost)
            reader_times.append(batch_reader_end - batch_start)
            batch_size = input_data_feed[4].shape[0]
            total_loss += loss * batch_size
            word_num = paddle.sum(input_data_feed[4])
            word_count += word_num.numpy()

            if batch_id > 0 and batch_id % 100 == 0:
                print(
                    "-- Epoch:[%d]; Batch:[%d]; ppl: %.5f, avg_batch_cost: %.5f s, avg_reader_cost: %.5f s"
                    % (epoch_id, batch_id, np.exp(
                        total_loss.numpy() / word_count), sum(batch_times) /
                       len(batch_times), sum(reader_times) / len(reader_times)))
                ce_ppl.append(np.exp(total_loss.numpy() / word_count))
                total_loss = 0.0
                word_count = 0.0
            batch_times, reader_times = [], []
            batch_start = time.time()

        train_epoch_cost = time.time() - epoch_start
        print(
            "\nTrain epoch:[%d]; epoch_cost: %.5f s; avg_batch_cost: %.5f s/step\n"
            % (epoch_id, train_epoch_cost, sum(epoch_times) / len(epoch_times)))
        ce_time.append(train_epoch_cost)

        path_name = os.path.join(args.model_path, "epoch_" + str(epoch_id))

        print("begin to save", path_name)
        paddle.save(model.state_dict(), path_name)
        print("save finished")
        dev_ppl = eval(model, val_loader)
        print("dev ppl", dev_ppl)
        test_ppl = eval(model, test_loader)
        print("test ppl", test_ppl)
        epoch_times = []

    if args.enable_ce:
        card_num = get_cards()
        _ppl = 0
        _time = 0
        try:
            _time = ce_time[-1]
            _ppl = ce_ppl[-1]
        except:
            print("ce info error")
        print("kpis\ttrain_duration_card%s\t%s" % (card_num, _time))
        print("kpis\ttrain_ppl_card%s\t%f" % (card_num, _ppl))


def get_cards():
    num = 0
    cards = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if cards != '':
        num = len(cards.split(","))
    return num


def eval(model, data_loader, epoch_id=0):
    model.eval()
    total_loss = 0.0
    word_count = 0.0
    for batch_id, input_data_feed in enumerate(data_loader):
        word_num = paddle.sum(input_data_feed[4])
        batch_size = input_data_feed[4].shape[0]
        word_count += word_num.numpy()

        loss = model(input_data_feed)

        total_loss += loss * batch_size
        word_count += word_num.numpy()

    ppl = np.exp(total_loss.numpy() / word_count)
    model.train()
    return ppl


if __name__ == '__main__':
    main()
