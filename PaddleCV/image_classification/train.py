#copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import time
import sys
import functools
import math


def set_paddle_flags(flags):
    for key, value in flags.items():
        if os.environ.get(key, None) is None:
            os.environ[key] = str(value)


# NOTE(paddle-dev): All of these flags should be
# set before `import paddle`. Otherwise, it would
# not take any effect. 
set_paddle_flags({
    'FLAGS_eager_delete_tensor_gb': 0,  # enable gc 
    'FLAGS_fraction_of_gpu_memory_to_use': 0.98
})

import argparse
import functools
import subprocess

import paddle
import paddle.fluid as fluid
import reader
from utils import *
import models
from build_model import create_model


def build_program(is_train, main_prog, startup_prog, args):
    """build program, and add grad op in program accroding to different mode

    Args:
        is_train: mode: train or test
        main_prog: main program
        startup_prog: strartup program
        args: arguments

    Returns : 
        train mode: [Loss, global_lr, py_reader]
        test mode: [Loss, py_reader]
    """
    model = models.__dict__[args.model]()
    with fluid.program_guard(main_prog, startup_prog):
        if args.random_seed:
            main_prog.random_seed = args.random_seed
            startup_prog.random_seed = args.random_seed
        with fluid.unique_name.guard():
            py_reader, loss_out = create_model(model, args, is_train)
            # add backward op in program
            if is_train:
                optimizer = create_optimizer(args)
                avg_cost = loss_out[0]
                optimizer.minimize(avg_cost)
                #XXX: fetch learning rate now, better implement is required here. 
                global_lr = optimizer._global_learning_rate()
                global_lr.persistable = True
                loss_out.append(global_lr)
            loss_out.append(py_reader)
    return loss_out


def train(args):
    """Train model
    
    Args:
        args: all arguments.    
    """
    startup_prog = fluid.Program()
    train_prog = fluid.Program()
    test_prog = fluid.Program()

    train_out = build_program(
        is_train=True,
        main_prog=train_prog,
        startup_prog=startup_prog,
        args=args)
    train_py_reader = train_out[-1]
    train_fetch_vars = train_out[:-1]
    train_fetch_list = []
    for var in train_fetch_vars:
        var.persistable = True
        train_fetch_list.append(var.name)

    test_out = build_program(
        is_train=False,
        main_prog=test_prog,
        startup_prog=startup_prog,
        args=args)
    test_py_reader = test_out[-1]
    test_fetch_vars = test_out[:-1]
    test_fetch_list = []
    for var in test_fetch_vars:
        var.persistable = True
        test_fetch_list.append(var.name)

    #Create test_prog and set layers' is_test params to True
    test_prog = test_prog.clone(for_test=True)

    gpu_id = int(os.environ.get('FLAGS_selected_gpus', 0))
    place = fluid.CUDAPlace(gpu_id) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(startup_prog)

    #init model by checkpoint or pretrianed model.
    init_model(exe, args, train_prog)

    train_reader = reader.train(settings=args)
    train_reader = paddle.batch(
        train_reader,
        batch_size=int(args.batch_size / fluid.core.get_cuda_device_count()),
        drop_last=True)

    test_reader = reader.val(settings=args)
    test_reader = paddle.batch(
        test_reader, batch_size=args.test_batch_size, drop_last=True)

    train_py_reader.decorate_sample_list_generator(train_reader, place)
    test_py_reader.decorate_sample_list_generator(test_reader, place)

    compiled_train_prog = best_strategy_compiled(args, train_prog,
                                                 train_fetch_vars[0])

    for pass_id in range(args.num_epochs):

        train_batch_id = 0
        test_batch_id = 0
        train_batch_time_record = []
        test_batch_time_record = []
        train_batch_metrics_record = []
        test_batch_metrics_record = []

        train_py_reader.start()

        try:
            while True:
                t1 = time.time()
                train_batch_metrics = exe.run(compiled_train_prog,
                                              fetch_list=train_fetch_list)
                t2 = time.time()
                train_batch_elapse = t2 - t1
                train_batch_time_record.append(train_batch_elapse)
                train_batch_metrics_avg = np.mean(
                    np.array(train_batch_metrics), axis=1)
                train_batch_metrics_record.append(train_batch_metrics_avg)

                print_info(pass_id, train_batch_id, args.print_step,
                           train_batch_metrics_avg, train_batch_elapse, "batch")
                sys.stdout.flush()
                train_batch_id += 1

        except fluid.core.EOFException:
            train_py_reader.reset()

        test_py_reader.start()
        try:
            while True:
                t1 = time.time()
                test_batch_metrics = exe.run(program=test_prog,
                                             fetch_list=test_fetch_list)
                t2 = time.time()
                test_batch_elapse = t2 - t1
                test_batch_time_record.append(test_batch_elapse)

                test_batch_metrics_avg = np.mean(
                    np.array(test_batch_metrics), axis=1)
                test_batch_metrics_record.append(test_batch_metrics_avg)

                print_info(pass_id, test_batch_id, args.print_step,
                           test_batch_metrics_avg, test_batch_elapse, "batch")
                sys.stdout.flush()
                test_batch_id += 1

        except fluid.core.EOFException:
            test_py_reader.reset()
        train_epoch_time_avg = np.mean(np.array(train_batch_time_record))
        train_epoch_metrics_avg = np.mean(
            np.array(train_batch_metrics_record), axis=0)

        test_epoch_time_avg = np.mean(np.array(test_batch_time_record))
        test_epoch_metrics_avg = np.mean(
            np.array(test_batch_metrics_record), axis=0)

        print_info(pass_id, 0, 0,
                   list(train_epoch_metrics_avg) + list(test_epoch_metrics_avg),
                   0, "epoch")
        #For now, save model per epoch. 
        save_model(args, exe, train_prog, pass_id)


def main():
    args = parse_args()
    print_arguments(args)
    check_args(args)
    train(args)


if __name__ == '__main__':
    main()
