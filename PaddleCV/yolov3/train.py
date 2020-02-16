#  Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
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
    'FLAGS_eager_delete_tensor_gb': 0,  # enable gc
    'FLAGS_memory_fraction_of_eager_deletion': 1,
    'FLAGS_fraction_of_gpu_memory_to_use': 0.98
})

import sys
import numpy as np
import random
import time
import shutil
import subprocess
from utility import (parse_args, print_arguments, 
                     SmoothedValue, check_gpu)

import paddle
import paddle.fluid as fluid
from paddle.fluid import profiler
import reader
from models.yolov3 import YOLOv3
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

    # check if set use_gpu=True in paddlepaddle cpu version
    check_gpu(cfg.use_gpu)

    if cfg.debug or args.enable_ce:
        fluid.default_startup_program().random_seed = 1000
        fluid.default_main_program().random_seed = 1000
        random.seed(0)
        np.random.seed(0)

    if not os.path.exists(cfg.model_save_dir):
        os.makedirs(cfg.model_save_dir)

    model = YOLOv3()
    model.build_model()
    input_size = cfg.input_size
    loss = model.loss()
    loss.persistable = True

    devices_num = get_device_num() if cfg.use_gpu else 1
    print("Found {} CUDA/CPU devices.".format(devices_num))

    learning_rate = cfg.learning_rate
    boundaries = cfg.lr_steps
    gamma = cfg.lr_gamma
    step_num = len(cfg.lr_steps)
    values = [learning_rate * (gamma**i) for i in range(step_num + 1)]

    optimizer = fluid.optimizer.Momentum(
        learning_rate=exponential_with_warmup_decay(
            learning_rate=learning_rate,
            boundaries=boundaries,
            values=values,
            warmup_iter=cfg.warm_up_iter,
            warmup_factor=cfg.warm_up_factor),
        regularization=fluid.regularizer.L2Decay(cfg.weight_decay),
        momentum=cfg.momentum)
    optimizer.minimize(loss)

    gpu_id = int(os.environ.get('FLAGS_selected_gpus', 0))
    place = fluid.CUDAPlace(gpu_id) if cfg.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    if cfg.pretrain:
        if not os.path.exists(cfg.pretrain):
            print("Pretrain weights not found: {}".format(cfg.pretrain))

        def if_exist(var):
            return os.path.exists(os.path.join(cfg.pretrain, var.name))

        fluid.io.load_vars(exe, cfg.pretrain, predicate=if_exist)

    build_strategy = fluid.BuildStrategy()
    build_strategy.memory_optimize = False  #gc and memory optimize may conflict
    syncbn = cfg.syncbn
    if (syncbn and devices_num <= 1) or num_trainers > 1:
        print("Disable syncbn in single device")
        syncbn = False
    build_strategy.sync_batch_norm = syncbn

    exec_strategy = fluid.ExecutionStrategy()
    if cfg.use_gpu and num_trainers > 1:
        dist_utils.prepare_for_multi_process(exe, build_strategy,
                                             fluid.default_main_program())
        exec_strategy.num_threads = 1

    compile_program = fluid.compiler.CompiledProgram(fluid.default_main_program(
    )).with_data_parallel(
        loss_name=loss.name,
        build_strategy=build_strategy,
        exec_strategy=exec_strategy)

    random_sizes = [cfg.input_size]
    if cfg.random_shape:
        random_sizes = [32 * i for i in range(10, 20)]

    total_iter = cfg.max_iter - cfg.start_iter
    mixup_iter = total_iter - cfg.no_mixup_iter

    shuffle = True
    if args.enable_ce:
        shuffle = False
    shuffle_seed = None
    # NOTE: yolov3 is a special model, if num_trainers > 1, each process
    # trian the completed dataset.
    # if num_trainers > 1: shuffle_seed  = 1
    train_reader = reader.train(
        input_size,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        shuffle_seed=shuffle_seed,
        total_iter=total_iter * devices_num,
        mixup_iter=mixup_iter * devices_num,
        random_sizes=random_sizes,
        use_multiprocess_reader=cfg.use_multiprocess_reader,
        num_workers=cfg.worker_num)
    py_reader = model.py_reader
    py_reader.decorate_paddle_reader(train_reader)

    def save_model(postfix):
        model_path = os.path.join(cfg.model_save_dir, postfix)
        if os.path.isdir(model_path):
            shutil.rmtree(model_path)
        fluid.io.save_persistables(exe, model_path)

    fetch_list = [loss]

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
            losses = exe.run(compile_program,
                             fetch_list=[v.name for v in fetch_list])
            smoothed_loss.add_value(np.mean(np.array(losses[0])))
            snapshot_loss += np.mean(np.array(losses[0]))
            snapshot_time += start_time - prev_start_time
            lr = np.array(fluid.global_scope().find_var('learning_rate')
                          .get_tensor())
            print("Iter {:d}, lr {:.6f}, loss {:.6f}, time {:.5f}".format(
                iter_id, lr[0],
                smoothed_loss.get_mean_value(), start_time - prev_start_time))
            sys.stdout.flush()
            #add profiler tools
            if args.is_profiler and iter_id == 5:
                profiler.start_profiler("All")
            elif args.is_profiler and iter_id == 10:
                profiler.stop_profiler("total", args.profiler_path)
                return

            if (iter_id + 1) % cfg.snapshot_iter == 0:
                save_model("model_iter{}".format(iter_id))
                print("Snapshot {} saved, average loss: {}, \
                      average time: {}".format(
                    iter_id + 1, snapshot_loss / float(cfg.snapshot_iter),
                    snapshot_time / float(cfg.snapshot_iter)))
                if args.enable_ce and iter_id == cfg.max_iter - 1:
                    if devices_num == 1:
                        print("kpis\ttrain_cost_1card\t%f" %
                              (snapshot_loss / float(cfg.snapshot_iter)))
                        print("kpis\ttrain_duration_1card\t%f" %
                              (snapshot_time / float(cfg.snapshot_iter)))
                    else:
                        print("kpis\ttrain_cost_8card\t%f" %
                              (snapshot_loss / float(cfg.snapshot_iter)))
                        print("kpis\ttrain_duration_8card\t%f" %
                              (snapshot_time / float(cfg.snapshot_iter)))

                snapshot_loss = 0
                snapshot_time = 0
    except fluid.core.EOFException:
        py_reader.reset()

    save_model('model_final')


if __name__ == '__main__':
    args = parse_args()
    print_arguments(args)
    train()
