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
import time
import shutil
from utility import parse_args, print_arguments, SmoothedValue, TrainingStats, now_time
import collections

import paddle
import paddle.fluid as fluid
import reader
import models.model_builder as model_builder
import models.resnet as resnet
from learning_rate import exponential_with_warmup_decay
from config import cfg


def train():
    learning_rate = cfg.learning_rate
    image_shape = [3, cfg.TRAIN.max_size, cfg.TRAIN.max_size]

    if cfg.debug:
        fluid.default_startup_program().random_seed = 1000
        fluid.default_main_program().random_seed = 1000
        import random
        random.seed(0)
        np.random.seed(0)

    devices = os.getenv("CUDA_VISIBLE_DEVICES") or ""
    devices_num = len(devices.split(","))
    total_batch_size = devices_num * cfg.TRAIN.im_per_batch

    model = model_builder.FasterRCNN(
        add_conv_body_func=resnet.add_ResNet50_conv4_body,
        add_roi_box_head_func=resnet.add_ResNet_roi_conv5_head,
        use_pyreader=cfg.use_pyreader,
        use_random=True)
    model.build_model(image_shape)
    losses, keys = model.loss()
    loss = losses[0]
    fetch_list = losses

    boundaries = cfg.lr_steps
    gamma = cfg.lr_gamma
    step_num = len(cfg.lr_steps)
    values = [learning_rate * (gamma**i) for i in range(step_num + 1)]

    lr = exponential_with_warmup_decay(
        learning_rate=learning_rate,
        boundaries=boundaries,
        values=values,
        warmup_iter=cfg.warm_up_iter,
        warmup_factor=cfg.warm_up_factor)
    optimizer = fluid.optimizer.Momentum(
        learning_rate=lr,
        regularization=fluid.regularizer.L2Decay(cfg.weight_decay),
        momentum=cfg.momentum)
    optimizer.minimize(loss)
    fetch_list = fetch_list + [lr]

    fluid.memory_optimize(
        fluid.default_main_program(), skip_opt_set=set(fetch_list))

    place = fluid.CUDAPlace(0) if cfg.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    if cfg.pretrained_model:

        def if_exist(var):
            return os.path.exists(os.path.join(cfg.pretrained_model, var.name))

        fluid.io.load_vars(exe, cfg.pretrained_model, predicate=if_exist)

    if cfg.parallel:
        exec_strategy = fluid.ExecutionStrategy()
        #exec_strategy.num_threads = 64
        train_exe = fluid.ParallelExecutor(
            use_cuda=bool(cfg.use_gpu),
            loss_name=loss.name,
            exec_strategy=exec_strategy)

    if cfg.use_pyreader:
        train_reader = reader.train(
            batch_size=cfg.TRAIN.im_per_batch,
            total_batch_size=total_batch_size,
            padding_total=cfg.TRAIN.padding_minibatch,
            shuffle=True)
        py_reader = model.py_reader
        py_reader.decorate_paddle_reader(train_reader)
    else:
        train_reader = reader.train(batch_size=total_batch_size, shuffle=True)
        feeder = fluid.DataFeeder(place=place, feed_list=model.feeds())

    def save_model(postfix):
        model_path = os.path.join(cfg.model_save_dir, postfix)
        if os.path.isdir(model_path):
            shutil.rmtree(model_path)
        fluid.io.save_persistables(exe, model_path)

    def train_loop_pyreader():
        py_reader.start()
        train_stats = TrainingStats(cfg.log_window, keys)
        try:
            start_time = time.time()
            prev_start_time = start_time
            for iter_id in range(cfg.max_iter):
                prev_start_time = start_time
                start_time = time.time()
                outs = train_exe.run(fetch_list=[v.name for v in fetch_list])
                stats = {k: np.array(v).mean() for k, v in zip(keys, outs[:-1])}
                train_stats.update(stats)
                logs = train_stats.log()
                strs = '{}, lr: {:.5f}, {}, time: {:.3f}'.format(
                    now_time(),
                    np.mean(outs[-1]), logs, start_time - prev_start_time)
                print(strs)
                sys.stdout.flush()
                if (iter_id + 1) % cfg.TRAIN.snapshot_iter == 0:
                    save_model("model_iter{}".format(iter_id))
        except (StopIteration, fluid.core.EOFException):
            py_reader.reset()

    def train_loop():
        start_time = time.time()
        prev_start_time = start_time
        start = start_time
        train_stats = TrainingStats(cfg.log_window, keys)
        for iter_id, data in enumerate(train_reader()):
            prev_start_time = start_time
            start_time = time.time()
            outs = train_exe.run(fetch_list=[v.name for v in fetch_list],
                                 feed=feeder.feed(data))
            stats = {k: np.array(v).mean() for k, v in zip(keys, outs[:-1])}
            train_stats.update(stats)
            logs = train_stats.log()
            strs = '{}, lr: {:.5f}, {}, time: {:.3f}'.format(
                now_time(),
                np.mean(outs[-1]), logs, start_time - prev_start_time)
            print(strs)
            sys.stdout.flush()
            if (iter_id + 1) % cfg.TRAIN.snapshot_iter == 0:
                save_model("model_iter{}".format(iter_id))
            if (iter_id + 1) == cfg.max_iter:
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
