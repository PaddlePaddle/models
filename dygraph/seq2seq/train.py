# -*- coding: utf-8 -*-
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
import logging
import random
import math
import contextlib

import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph_grad_clip import GradClipByGlobalNorm

import reader

import sys
if sys.version[0] == '2':
    reload(sys)
    sys.setdefaultencoding("utf-8")

from args import *
from base_model import BaseModel
from attention_model import AttentionModel
import logging
import pickle


def main():
    args = parse_args()
    print(args)
    num_layers = args.num_layers
    src_vocab_size = args.src_vocab_size
    tar_vocab_size = args.tar_vocab_size
    batch_size = args.batch_size
    dropout = args.dropout
    init_scale = args.init_scale
    max_grad_norm = args.max_grad_norm
    hidden_size = args.hidden_size

    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    with fluid.dygraph.guard(place):
        args.enable_ce = True
        if args.enable_ce:
            fluid.default_startup_program().random_seed = 102
            fluid.default_main_program().random_seed = 102
            np.random.seed(102)
            random.seed(102)

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
        gloabl_norm_clip = GradClipByGlobalNorm(max_grad_norm)
        lr = args.learning_rate
        opt_type = args.optimizer
        if opt_type == "sgd":
            optimizer = fluid.optimizer.SGD(lr, parameter_list=model.parameters())
        elif opt_type == "adam":
            optimizer = fluid.optimizer.Adam(lr, parameter_list=model.parameters())
        else:
            print("only support [sgd|adam]")
            raise Exception("opt type not support")

        train_data_prefix = args.train_data_prefix
        eval_data_prefix = args.eval_data_prefix
        test_data_prefix = args.test_data_prefix
        vocab_prefix = args.vocab_prefix
        src_lang = args.src_lang
        tar_lang = args.tar_lang
        print("begin to load data")
        train_data_iter = fluid.io.DataLoader.from_generator(capacity=32, use_double_buffer=True, iterable=True, return_list=True, use_multiprocess=True)
        valid_data_iter = fluid.io.DataLoader.from_generator(capacity=32, use_double_buffer=True, iterable=True, return_list=True, use_multiprocess=True)
        test_data_iter = fluid.io.DataLoader.from_generator(capacity=32, use_double_buffer=True, iterable=True, return_list=True, use_multiprocess=True)

        train_reader = reader.get_reader(batch_size, src_lang, tar_lang, vocab_prefix, train_data_prefix, max_len=args.max_len, reader_mode='train', enable_ce=args.enable_ce, cache_num=20)
        train_data_iter.set_batch_generator(train_reader, place)
        valid_reader = reader.get_reader(batch_size, src_lang, tar_lang, vocab_prefix, eval_data_prefix, reader_mode='valid', mode='eval', cache_num=20)
        valid_data_iter.set_batch_generator(valid_reader, place)
        test_reader = reader.get_reader(batch_size, src_lang, tar_lang, vocab_prefix, test_data_prefix, reader_mode='test', mode='eval', cache_num=20)
        test_data_iter.set_batch_generator(test_reader, place)
        print("finished load data")

        # get train epoch size
        def eval(eval_data_iter, epoch_id=0):
            model.eval()
            total_loss = 0.0
            word_count = 0.0
            for batch_id, batch in enumerate(eval_data_iter):
                input_data_feed1, input_data_feed2, input_data_feed3, input_data_feed4, input_data_feed5,  word_num = batch
                input_data_feed = [input_data_feed1, input_data_feed2, input_data_feed3, input_data_feed4, input_data_feed5]
                loss = model(input_data_feed)

                total_loss += loss * batch_size
                word_count += word_num
            ppl = np.exp(total_loss.numpy() / word_count.numpy())
            model.train()
            return ppl

        max_epoch = args.max_epoch
        for epoch_id in range(max_epoch):
            model.train()
            start_time = time.time()

            total_loss = 0
            word_count = 0.0
            batch_times = []
            for batch_id, batch in enumerate(train_data_iter):
                batch_start_time = time.time()
                input_data_feed1, input_data_feed2, input_data_feed3, input_data_feed4, input_data_feed5, word_num = batch
                input_data_feed = [input_data_feed1, input_data_feed2, input_data_feed3, input_data_feed4, input_data_feed5]
                word_count += word_num
                loss = model(input_data_feed)
                # print(loss.numpy()[0])
                loss.backward()
                optimizer.minimize(loss, grad_clip = gloabl_norm_clip)
                model.clear_gradients()
                total_loss += loss * batch_size
                batch_end_time = time.time()
                batch_time = batch_end_time - batch_start_time
                batch_times.append(batch_time)

                if batch_id > 0 and batch_id % 100 == 0:
                    print("-- Epoch:[%d]; Batch:[%d]; Time: %.5f s; ppl: %.5f" %
                        (epoch_id, batch_id, batch_time,
                        np.exp(total_loss.numpy() / word_count.numpy())))
                    total_loss = 0.0
                    word_count = 0.0

            end_time = time.time()
            epoch_time = end_time - start_time
            print(
                "\nTrain epoch:[%d]; Epoch Time: %.5f; avg_time: %.5f s/step\n"
                % (epoch_id, epoch_time, sum(batch_times) / len(batch_times)))

            
            dir_name = os.path.join(args.model_path,
                                    "epoch_" + str(epoch_id))
            print("begin to save", dir_name)
            paddle.fluid.save_dygraph(model.state_dict(), dir_name)
            print("save finished")
            dev_ppl = eval(valid_data_iter)
            print("dev ppl", dev_ppl)
            test_ppl = eval(test_data_iter)
            print("test ppl", test_ppl)


def get_cards():
    num = 0
    cards = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if cards != '':
        num = len(cards.split(","))
    return num


def check_version():
    """
    Log error and exit when the installed version of paddlepaddle is
    not satisfied.
    """
    err = "PaddlePaddle version 1.6 or higher is required, " \
          "or a suitable develop version is satisfied as well. \n" \
          "Please make sure the version is good with your code." \

    try:
        fluid.require_version('1.6.0')
    except Exception as e:
        logger.error(err)
        sys.exit(1)


if __name__ == '__main__':
    check_version()
    main()
