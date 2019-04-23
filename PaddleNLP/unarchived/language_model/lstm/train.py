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
sys.path.append('..')
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from args import *
import lm_model
import logging
import pickle

SEED = 123


def get_current_model_para(train_prog, train_exe):
    param_list = train_prog.block(0).all_parameters()
    param_name_list = [p.name for p in param_list]

    vals = {}
    for p_name in param_name_list:
        p_array = np.array(fluid.global_scope().find_var(p_name).get_tensor())
        vals[p_name] = p_array

    return vals


def save_para_npz(train_prog, train_exe):
    print("begin to save model to model_base")
    param_list = train_prog.block(0).all_parameters()
    param_name_list = [p.name for p in param_list]

    vals = {}
    for p_name in param_name_list:
        p_array = np.array(fluid.global_scope().find_var(p_name).get_tensor())
        vals[p_name] = p_array

    emb = vals["embedding_para"]
    print("begin to save model to model_base")
    np.savez("mode_base", **vals)


def train():
    args = parse_args()
    model_type = args.model_type
    rnn_model = args.rnn_model
    logger = logging.getLogger("lm")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    if args.enable_ce:
        fluid.default_startup_program().random_seed = SEED
    if args.log_path:
        file_handler = logging.FileHandler(args.log_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    else:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    logger.info('Running with args : {}'.format(args))

    vocab_size = 10000
    if model_type == "test":
        num_layers = 1
        batch_size = 2
        hidden_size = 10
        num_steps = 3
        init_scale = 0.1
        max_grad_norm = 5.0
        epoch_start_decay = 1
        max_epoch = 1
        dropout = 0.0
        lr_decay = 0.5
        base_learning_rate = 1.0
    elif model_type == "small":
        num_layers = 2
        batch_size = 20
        hidden_size = 200
        num_steps = 20
        init_scale = 0.1
        max_grad_norm = 5.0
        epoch_start_decay = 4
        max_epoch = 13
        dropout = 0.0
        lr_decay = 0.5
        base_learning_rate = 1.0
    elif model_type == "medium":
        num_layers = 2
        batch_size = 20
        hidden_size = 650
        num_steps = 35
        init_scale = 0.05
        max_grad_norm = 5.0
        epoch_start_decay = 6
        max_epoch = 39
        dropout = 0.5
        lr_decay = 0.8
        base_learning_rate = 1.0
    elif model_type == "large":
        num_layers = 2
        batch_size = 20
        hidden_size = 1500
        num_steps = 35
        init_scale = 0.04
        max_grad_norm = 10.0
        epoch_start_decay = 14
        max_epoch = 55
        dropout = 0.65
        lr_decay = 1.0 / 1.15
        base_learning_rate = 1.0
    else:
        print("model type not support")
        return

    # Training process
    loss, last_hidden, last_cell, feed_order = lm_model.lm_model(
        hidden_size,
        vocab_size,
        batch_size,
        num_layers=num_layers,
        num_steps=num_steps,
        init_scale=init_scale,
        dropout=dropout,
        rnn_model=rnn_model)
    # clone from default main program and use it as the validation program
    main_program = fluid.default_main_program()
    inference_program = fluid.default_main_program().clone(for_test=True)

    fluid.clip.set_gradient_clip(clip=fluid.clip.GradientClipByGlobalNorm(
        clip_norm=max_grad_norm))

    learning_rate = fluid.layers.create_global_var(
        name="learning_rate",
        shape=[1],
        value=1.0,
        dtype='float32',
        persistable=True)

    optimizer = fluid.optimizer.SGD(learning_rate=learning_rate)

    optimizer.minimize(loss)

    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = Executor(place)
    exe.run(framework.default_startup_program())

    data_path = args.data_path
    print("begin to load data")
    raw_data = reader.ptb_raw_data(data_path)
    print("finished load data")
    train_data, valid_data, test_data, _ = raw_data

    def prepare_input(batch, init_hidden, init_cell, epoch_id=0, with_lr=True):
        x, y = batch
        new_lr = base_learning_rate * (lr_decay**max(
            epoch_id + 1 - epoch_start_decay, 0.0))
        lr = np.ones((1), dtype='float32') * new_lr
        res = {}
        x = x.reshape((-1, num_steps, 1))
        y = y.reshape((-1, 1))

        res['x'] = x
        res['y'] = y
        res['init_hidden'] = init_hidden
        res['init_cell'] = init_cell
        if with_lr:
            res['learning_rate'] = lr

        return res

    def eval(data):
        # when eval the batch_size set to 1
        eval_data_iter = reader.get_data_iter(data, batch_size, num_steps)
        total_loss = 0.0
        iters = 0
        init_hidden = np.zeros(
            (num_layers, batch_size, hidden_size), dtype='float32')
        init_cell = np.zeros(
            (num_layers, batch_size, hidden_size), dtype='float32')
        for batch_id, batch in enumerate(eval_data_iter):
            input_data_feed = prepare_input(
                batch, init_hidden, init_cell, epoch_id, with_lr=False)
            fetch_outs = exe.run(
                inference_program,
                feed=input_data_feed,
                fetch_list=[loss.name, last_hidden.name, last_cell.name],
                use_program_cache=True)

            cost_train = np.array(fetch_outs[0])
            init_hidden = np.array(fetch_outs[1])
            init_cell = np.array(fetch_outs[2])

            total_loss += cost_train
            iters += num_steps

        ppl = np.exp(total_loss / iters)
        return ppl

    # get train epoch size
    batch_len = len(train_data) // batch_size
    epoch_size = (batch_len - 1) // num_steps
    log_interval = epoch_size // 10
    total_time = 0.0
    for epoch_id in range(max_epoch):
        start_time = time.time()
        print("epoch id", epoch_id)
        train_data_iter = reader.get_data_iter(train_data, batch_size,
                                               num_steps)

        total_loss = 0

        init_hidden = None
        init_cell = None
        #debug_para(fluid.framework.default_main_program(), parallel_executor)
        total_loss = 0
        iters = 0
        init_hidden = np.zeros(
            (num_layers, batch_size, hidden_size), dtype='float32')
        init_cell = np.zeros(
            (num_layers, batch_size, hidden_size), dtype='float32')
        for batch_id, batch in enumerate(train_data_iter):
            input_data_feed = prepare_input(
                batch, init_hidden, init_cell, epoch_id=epoch_id)
            fetch_outs = exe.run(feed=input_data_feed,
                                 fetch_list=[
                                     loss.name, last_hidden.name,
                                     last_cell.name, 'learning_rate'
                                 ],
                                 use_program_cache=True)

            cost_train = np.array(fetch_outs[0])
            init_hidden = np.array(fetch_outs[1])
            init_cell = np.array(fetch_outs[2])

            lr = np.array(fetch_outs[3])

            total_loss += cost_train
            iters += num_steps
            if batch_id > 0 and batch_id % log_interval == 0:
                ppl = np.exp(total_loss / iters)
                print("ppl ", batch_id, ppl[0], lr[0])

        ppl = np.exp(total_loss / iters)
        if epoch_id == 0 and ppl[0] > 1000:
            # for bad init, after first epoch, the loss is over 1000
            # no more need to continue
            return
        end_time = time.time()
        total_time += end_time - start_time
        print("train ppl", ppl[0])

        if epoch_id == max_epoch - 1 and args.enable_ce:
            print("ptblm\tlstm_language_model_duration\t%s" %
                  (total_time / max_epoch))
            print("ptblm\tlstm_language_model_loss\t%s" % ppl[0])

        model_path = os.path.join("model_new/", str(epoch_id))
        if not os.path.isdir(model_path):
            os.makedirs(model_path)
        fluid.io.save_persistables(
            executor=exe, dirname=model_path, main_program=main_program)
        valid_ppl = eval(valid_data)
        print("valid ppl", valid_ppl[0])
    test_ppl = eval(test_data)
    print("test ppl", test_ppl[0])


if __name__ == '__main__':
    train()
