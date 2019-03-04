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

import os
import time
import numpy as np
import argparse
from utility import parse_args, add_arguments, print_arguments

import paddle
import paddle.fluid as fluid
import reader
import paddle.fluid.profiler as profiler

import models.model_builder as model_builder
import models.resnet as resnet
from learning_rate import exponential_with_warmup_decay
from config import cfg


def train():
    learning_rate = cfg.learning_rate
    image_shape = [3, cfg.TRAIN.max_size, cfg.TRAIN.max_size]
    num_iterations = cfg.max_iter

    devices = os.getenv("CUDA_VISIBLE_DEVICES") or ""
    devices_num = len(devices.split(","))
    total_batch_size = devices_num * cfg.TRAIN.im_per_batch
    model = model_builder.RCNN(
        add_conv_body_func=resnet.add_ResNet50_conv4_body,
        add_roi_box_head_func=resnet.add_ResNet_roi_conv5_head,
        use_pyreader=cfg.use_pyreader,
        use_random=False)
    model.build_model(image_shape)
    losses, keys = model.loss()
    loss = losses[0]
    fetch_list = [loss]

    boundaries = cfg.lr_steps
    gamma = cfg.lr_gamma
    step_num = len(cfg.lr_steps)
    values = [learning_rate * (gamma**i) for i in range(step_num + 1)]

    optimizer = fluid.optimizer.Momentum(
        learning_rate=exponential_with_warmup_decay(
            learning_rate=learning_rate,
            boundaries=boundaries,
            values=values,
            warmup_iter=500,
            warmup_factor=1.0 / 3.0),
        regularization=fluid.regularizer.L2Decay(0.0001),
        momentum=0.9)
    optimizer.minimize(loss)

    fluid.memory_optimize(fluid.default_main_program())

    place = fluid.CUDAPlace(0) if cfg.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    if cfg.pretrained_model:

        def if_exist(var):
            return os.path.exists(os.path.join(cfg.pretrained_model, var.name))

        fluid.io.load_vars(exe, cfg.pretrained_model, predicate=if_exist)

    if cfg.parallel:
        train_exe = fluid.ParallelExecutor(
            use_cuda=bool(cfg.use_gpu), loss_name=loss.name)

    if cfg.use_pyreader:
        train_reader = reader.train(
            batch_size=cfg.TRAIN.im_per_batch,
            total_batch_size=total_batch_size,
            padding_total=cfg.TRAIN.padding_minibatch,
            shuffle=False)
        py_reader = model.py_reader
        py_reader.decorate_paddle_reader(train_reader)
    else:
        train_reader = reader.train(batch_size=total_batch_size, shuffle=False)
        feeder = fluid.DataFeeder(place=place, feed_list=model.feeds())

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
                outs = train_exe.run(fetch_list=[v.name for v in fetch_list],
                                     feed=feeder.feed(data))
            else:
                outs = exe.run(fluid.default_main_program(),
                               fetch_list=[v.name for v in fetch_list],
                               feed=feeder.feed(data))
            end_time = time.time()
            run_time.append(end_time - start_time)
            total_images += len(data)
            print("Batch {:d}, loss {:.6f} ".format(batch_id, np.mean(outs[0])))
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
                    outs = train_exe.run(
                        fetch_list=[v.name for v in fetch_list])
                else:
                    outs = exe.run(fluid.default_main_program(),
                                   fetch_list=[v.name for v in fetch_list])
                end_time = time.time()
                run_time.append(end_time - start_time)
                total_images += devices_num
                print("Batch {:d}, loss {:.6f} ".format(batch_id,
                                                        np.mean(outs[0])))
        except fluid.core.EOFException:
            py_reader.reset()

        return reader_time, run_time, total_images

    run_func = run if not cfg.use_pyreader else run_pyreader

    # warm-up
    run_func(2)
    # profiling
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
