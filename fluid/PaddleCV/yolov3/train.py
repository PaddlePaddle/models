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
import models
from learning_rate import exponential_with_warmup_decay
from config.config import cfg


def train():

    if cfg.debug:
        fluid.default_startup_program().random_seed = 1000
        fluid.default_main_program().random_seed = 1000
        random.seed(0)
        np.random.seed(0)
        
    if not os.path.exists(cfg.model_save_dir):
        os.makedirs(cfg.model_save_dir)

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

    random_sizes = []
    if cfg.random_shape:
        random_sizes = [32 * i for i in range(10, 20)]

    mixup_iter = cfg.max_iter - cfg.start_iter - cfg.no_mixup_iter
    if cfg.use_pyreader:
        train_reader = reader.train(input_size, batch_size=int(hyperparams['batch'])/devices_num, shuffle=True, mixup_iter=mixup_iter, random_sizes=random_sizes, use_multiprocessing=cfg.use_multiprocess)
        py_reader = model.py_reader
        py_reader.decorate_paddle_reader(train_reader)
    else:
        train_reader = reader.train(input_size, batch_size=int(hyperparams['batch']), shuffle=True, mixup_iter=mixup_iter, random_sizes=random_sizes, use_multiprocessing=cfg.use_multiprocess)
        feeder = fluid.DataFeeder(place=place, feed_list=model.feeds())

    def save_model(postfix):
        model_path = os.path.join(cfg.model_save_dir, postfix)
        if os.path.isdir(model_path):
            shutil.rmtree(model_path)
        fluid.io.save_persistables(base_exe, model_path)

    fetch_list = [loss]

    def train_loop_pyreader():
        py_reader.start()
        smoothed_loss = SmoothedValue()
        try:
            start_time = time.time()
            prev_start_time = start_time
            snapshot_loss = 0
            snapshot_time = 0
            for iter_id in range(cfg.start_iter, cfg.max_iter):
                prev_start_time = start_time
                start_time = time.time()
                losses = exe.run(fetch_list=[v.name for v in fetch_list])
                smoothed_loss.add_value(np.mean(np.array(losses[0])))
                snapshot_loss += np.mean(np.array(losses[0]))
                snapshot_time += start_time - prev_start_time
                lr = np.array(fluid.global_scope().find_var('learning_rate')
                              .get_tensor())
                print("Iter {:d}, lr {:.6f}, loss {:.6f}, time {:.5f}".format(
                    iter_id, lr[0],
                    smoothed_loss.get_mean_value(), start_time - prev_start_time))
                sys.stdout.flush()
                if (iter_id + 1) % cfg.snapshot_iter == 0:
                    save_model("model_iter{}".format(iter_id))
                    print("Snapshot {} saved, average loss: {}, average time: {}".format(
                        iter_id + 1, snapshot_loss / float(cfg.snapshot_iter), 
                        snapshot_time / float(cfg.snapshot_iter)))
                    snapshot_loss = 0
                    snapshot_time = 0
        except fluid.core.EOFException:
            py_reader.reset()

    def train_loop():
        start_time = time.time()
        prev_start_time = start_time
        start = start_time
        smoothed_loss = SmoothedValue()
        snapshot_loss = 0
        snapshot_time = 0
        for iter_id, data in enumerate(train_reader()):
            iter_id += cfg.start_iter
            prev_start_time = start_time
            start_time = time.time()
            losses = exe.run(fetch_list=[v.name for v in fetch_list],
                                   feed=feeder.feed(data))
            smoothed_loss.add_value(losses[0])
            snapshot_loss += losses[0]
            snapshot_time += start_time - prev_start_time
            lr = np.array(fluid.global_scope().find_var('learning_rate')
                          .get_tensor())
            print("Iter {:d}, lr: {:.6f}, loss: {:.4f}, time {:.5f}".format(
                iter_id, lr[0], smoothed_loss.get_mean_value(), start_time - prev_start_time))
            sys.stdout.flush()

            if (iter_id + 1) % cfg.snapshot_iter == 0:
                save_model("model_iter{}".format(iter_id))
                print("Snapshot {} saved, average loss: {}, average time: {}".format(
                    iter_id + 1, snapshot_loss / float(cfg.snapshot_iter), 
                    snapshot_time / float(cfg.snapshot_iter)))
                snapshot_loss = 0
                snapshot_time = 0
            if (iter_id + 1) == cfg.max_iter:
                print("Finish iter {}".format(iter_id))
                break

    if cfg.use_pyreader:
        train_loop_pyreader()
    else:
        train_loop()
    save_model('model_final')


if __name__ == '__main__':
    args = parse_args()
    print_arguments(args)
    train()
