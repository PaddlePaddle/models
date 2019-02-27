#  Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
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
import sys
import numpy as np
import random
import time
import shutil
from utility import parse_args, print_arguments, SmoothedValue

import paddle
import paddle.fluid as fluid
import reader
import paddle.fluid.profiler as profiler

import models
from learning_rate import exponential_with_warmup_decay
from config.config import cfg

def train():

    model = models.YOLOv3(cfg.model_cfg_path, use_pyreader=cfg.use_pyreader)
    model.build_model()
    input_size = model.get_input_size()
    loss = model.loss()
    loss.persistable = True

    hyperparams = model.get_hyperparams()
    devices = os.getenv("CUDA_VISIBLE_DEVICES") or ""
    devices_num = len(devices.split(","))
    print("Found {} CUDA devices.".format(devices_num))

    learning_rate = float(hyperparams['learning_rate'])
    num_iterations = cfg.max_iter
    boundaries = cfg.lr_steps
    gamma = cfg.lr_gamma
    step_num = len(cfg.lr_steps)
    if isinstance(gamma, list):
        values = [learning_rate * g for g in gamma]
    else:
        values = [learning_rate * (gamma**i) for i in range(step_num + 1)]

    optimizer = fluid.optimizer.Momentum(
        learning_rate=exponential_with_warmup_decay(
            learning_rate=learning_rate,
            boundaries=boundaries,
            values=values,
            warmup_iter=cfg.warm_up_iter,
            warmup_factor=cfg.warm_up_factor,
            start_step=cfg.start_iter),
        regularization=fluid.regularizer.L2Decay(float(hyperparams['decay'])),
        momentum=float(hyperparams['momentum']))
    optimizer.minimize(loss)

    fluid.memory_optimize(fluid.default_main_program())

    place = fluid.CUDAPlace(0) if cfg.use_gpu else fluid.CPUPlace()
    base_exe = fluid.Executor(place)
    base_exe.run(fluid.default_startup_program())

    if cfg.pretrain_base:
        def if_exist(var):
            return os.path.exists(os.path.join(cfg.pretrain_base, var.name))
        fluid.io.load_vars(base_exe, cfg.pretrain_base, predicate=if_exist)

    if cfg.parallel:
        exe = fluid.ParallelExecutor( use_cuda=bool(cfg.use_gpu), loss_name=loss.name)
    else:
        exe = base_exe

    random_sizes = [cfg.input_size]
    if cfg.random_shape:
        random_sizes = [32 * i for i in range(10, 20)]

    random_shape_iter = cfg.max_iter - cfg.start_iter - cfg.tune_iter
    if cfg.use_pyreader:
        train_reader = reader.train(input_size, batch_size=int(hyperparams['batch'])/devices_num, shuffle=True, random_shape_iter=random_shape_iter, random_sizes=random_sizes, interval=10, pyreader_num=devices_num,use_multiprocessing=cfg.use_multiprocess)
        py_reader = model.py_reader
        py_reader.decorate_paddle_reader(train_reader)
    else:
        train_reader = reader.train(input_size, batch_size=int(hyperparams['batch']), shuffle=True, random_shape_iter=random_shape_iter, random_sizes=random_sizes, interval=10,use_multiprocessing=cfg.use_multiprocess)
        feeder = fluid.DataFeeder(place=place, feed_list=model.feeds())

    fetch_list = [loss]

    def run(iterations):
        reader_time = []
        run_time = []
        total_images = 0

        for batch_id in range(iterations):
            start_time = time.time()
            data = next(train_reader())
            end_time = time.time()
            reader_time.append(end_time - start_time)
            start_time = time.time()
            if cfg.parallel:
                losses = exe.run(fetch_list=[v.name for v in fetch_list],
                                       feed=feeder.feed(data))
            else:
                losses = base_exe.run(fluid.default_main_program(),
                                 fetch_list=[v.name for v in fetch_list],
                                 feed=feeder.feed(data))
            end_time = time.time()
            run_time.append(end_time - start_time)
            total_images += len(data)

            lr = np.array(fluid.global_scope().find_var('learning_rate')
                          .get_tensor())
            print("Batch {:d}, lr {:.6f}, loss {:.6f} ".format(batch_id, lr[0],
                                                               losses[0][0]))
        return reader_time, run_time, total_images

    def run_pyreader(iterations):
        reader_time = [0]
        run_time = []
        total_images = 0

        py_reader.start()
        try:
            for batch_id in range(iterations):
                start_time = time.time()
                if cfg.parallel:
                    losses = exe.run(
                        fetch_list=[v.name for v in fetch_list])
                else:
                    losses = base_exe.run(fluid.default_main_program(),
                                     fetch_list=[v.name for v in fetch_list])
                end_time = time.time()
                run_time.append(end_time - start_time)
                total_images += devices_num
                lr = np.array(fluid.global_scope().find_var('learning_rate')
                              .get_tensor())
                print("Batch {:d}, lr {:.6f}, loss {:.6f} ".format(batch_id, lr[0],
                                                               losses[0][0]))
        except fluid.core.EOFException:
            py_reader.reset()

        return reader_time, run_time, total_images

    run_func = run if not cfg.use_pyreader else run_pyreader

    # warm-up
    run_func(2)

    #profiling
    start = time.time()
    if cfg.use_profile:
        with profiler.profiler('GPU', 'total', '/tmp/profile_file'):
            reader_time, run_time, total_images = run_func(num_iterations)
    else:
        reader_time, run_time, total_images = run_func(num_iterations)

    end = time.time()
    total_time = end - start
    print("Total time: {0}, reader time: {1} s, run time: {2} s, images/s: {3}".
          format(total_time,
                 np.sum(reader_time),
                 np.sum(run_time), total_images / total_time))

if __name__ == '__main__':
    args = parse_args()
    print_arguments(args)
    train()

