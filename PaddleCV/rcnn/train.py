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


def set_paddle_flags(flags):
    for key, value in flags.items():
        if os.environ.get(key, None) is None:
            os.environ[key] = str(value)


set_paddle_flags({
    'FLAGS_conv_workspace_size_limit': 500,
    'FLAGS_eager_delete_tensor_gb': 0,  # enable gc
    'FLAGS_memory_fraction_of_eager_deletion': 1,
    'FLAGS_fraction_of_gpu_memory_to_use': 0.98
})

import sys
import numpy as np
import time
import shutil
from utility import parse_args, print_arguments, SmoothedValue, TrainingStats, now_time, check_gpu
import collections

import paddle
import paddle.fluid as fluid
import reader
import models.model_builder as model_builder
import models.resnet as resnet
from learning_rate import exponential_with_warmup_decay
from config import cfg
import dist_utils

num_trainers = int(os.environ.get('PADDLE_TRAINERS_NUM', 1))


def get_device_num():
    # NOTE(zcd): for multi-processe training, each process use one GPU card.
    if num_trainers > 1:
        return 1
    return fluid.core.get_cuda_device_count()


def train():
    learning_rate = cfg.learning_rate
    image_shape = [3, cfg.TRAIN.max_size, cfg.TRAIN.max_size]

    if cfg.enable_ce:
        fluid.default_startup_program().random_seed = 1000
        fluid.default_main_program().random_seed = 1000
        import random
        random.seed(0)
        np.random.seed(0)

    devices_num = get_device_num()
    total_batch_size = devices_num * cfg.TRAIN.im_per_batch

    use_random = True
    if cfg.enable_ce:
        use_random = False
    model = model_builder.RCNN(
        add_conv_body_func=resnet.add_ResNet50_conv4_body,
        add_roi_box_head_func=resnet.add_ResNet_roi_conv5_head,
        use_pyreader=cfg.use_pyreader,
        use_random=use_random)
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

    for var in fetch_list:
        var.persistable = True

    #fluid.memory_optimize(fluid.default_main_program(), skip_opt_set=set(fetch_list))
    gpu_id = int(os.environ.get('FLAGS_selected_gpus', 0))
    place = fluid.CUDAPlace(gpu_id) if cfg.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    if cfg.pretrained_model:

        def if_exist(var):
            return os.path.exists(os.path.join(cfg.pretrained_model, var.name))

        fluid.io.load_vars(exe, cfg.pretrained_model, predicate=if_exist)

    if cfg.parallel:
        build_strategy = fluid.BuildStrategy()
        build_strategy.memory_optimize = False
        build_strategy.enable_inplace = True
        exec_strategy = fluid.ExecutionStrategy()
        exec_strategy.num_iteration_per_drop_scope = 10

        if num_trainers > 1 and cfg.use_gpu:
            dist_utils.prepare_for_multi_process(exe, build_strategy,
                                                 fluid.default_main_program())
            # NOTE: the process is fast when num_threads is 1 
            # for multi-process training.
            exec_strategy.num_threads = 1

        train_exe = fluid.ParallelExecutor(
            use_cuda=bool(cfg.use_gpu),
            loss_name=loss.name,
            build_strategy=build_strategy,
            exec_strategy=exec_strategy)
    else:
        train_exe = exe

    shuffle = True
    if cfg.enable_ce:
        shuffle = False
    # NOTE: do not shuffle dataset when using multi-process training 
    shuffle_seed = None
    if num_trainers > 1:
        shuffle_seed = 1
    if cfg.use_pyreader:
        train_reader = reader.train(
            batch_size=cfg.TRAIN.im_per_batch,
            total_batch_size=total_batch_size,
            padding_total=cfg.TRAIN.padding_minibatch,
            shuffle=shuffle,
            shuffle_seed=shuffle_seed)
        if num_trainers > 1:
            assert shuffle_seed is not None, \
                "If num_trainers > 1, the shuffle_seed must be set, because " \
                "the order of batch data generated by reader " \
                "must be the same in the respective processes."
            # NOTE: the order of batch data generated by batch_reader
            # must be the same in the respective processes.
            if num_trainers > 1:
                train_reader = fluid.contrib.reader.distributed_batch_reader(
                    train_reader)
        py_reader = model.py_reader
        py_reader.decorate_paddle_reader(train_reader)
    else:
        if num_trainers > 1: shuffle = False
        train_reader = reader.train(
            batch_size=total_batch_size, shuffle=shuffle)
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
                strs = '{}, iter: {}, lr: {:.5f}, {}, time: {:.3f}'.format(
                    now_time(), iter_id,
                    np.mean(outs[-1]), logs, start_time - prev_start_time)
                print(strs)
                sys.stdout.flush()
                if (iter_id + 1) % cfg.TRAIN.snapshot_iter == 0:
                    save_model("model_iter{}".format(iter_id))
            end_time = time.time()
            total_time = end_time - start_time
            last_loss = np.array(outs[0]).mean()
            if cfg.enable_ce:
                gpu_num = devices_num
                epoch_idx = iter_id + 1
                loss = last_loss
                print("kpis\teach_pass_duration_card%s\t%s" %
                      (gpu_num, total_time / epoch_idx))
                print("kpis\ttrain_loss_card%s\t%s" % (gpu_num, loss))
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
            strs = '{}, iter: {}, lr: {:.5f}, {}, time: {:.3f}'.format(
                now_time(), iter_id,
                np.mean(outs[-1]), logs, start_time - prev_start_time)
            print(strs)
            sys.stdout.flush()
            if (iter_id + 1) % cfg.TRAIN.snapshot_iter == 0:
                save_model("model_iter{}".format(iter_id))
            if (iter_id + 1) == cfg.max_iter:
                break
        end_time = time.time()
        total_time = end_time - start_time
        last_loss = np.array(outs[0]).mean()
        # only for ce
        if cfg.enable_ce:
            gpu_num = devices_num
            epoch_idx = iter_id + 1
            loss = last_loss
            print("kpis\teach_pass_duration_card%s\t%s" %
                  (gpu_num, total_time / epoch_idx))
            print("kpis\ttrain_loss_card%s\t%s" % (gpu_num, loss))

    if cfg.use_pyreader:
        train_loop_pyreader()
    else:
        train_loop()
    save_model('model_final')


if __name__ == '__main__':
    args = parse_args()
    print_arguments(args)
    check_gpu(args.use_gpu)
    train()
