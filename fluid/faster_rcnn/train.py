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
from utility import parse_args, print_arguments, SmoothedValue

import paddle
import paddle.fluid as fluid
import reader
import models.model_builder as model_builder
import models.resnet as resnet
from learning_rate import exponential_with_warmup_decay
from config import *


def train(args):
    print('====train===')
    print(TrainConfig.rpn_nms_thresh)
    print('====infer===')
    print(InferConfig.rpn_nms_thresh)
    learning_rate = SolverConfig.learning_rate
    image_shape = [3, TrainConfig.max_size, TrainConfig.max_size]

    if args.debug:
        fluid.default_startup_program().random_seed = 1000
        fluid.default_main_program().random_seed = 1000
        import random
        random.seed(0)
        np.random.seed(0)

    devices = os.getenv("CUDA_VISIBLE_DEVICES") or ""
    devices_num = len(devices.split(","))
    total_batch_size = devices_num * TrainConfig.im_per_batch

    model = model_builder.FasterRCNN(
        add_conv_body_func=resnet.add_ResNet50_conv4_body,
        add_roi_box_head_func=resnet.add_ResNet_roi_conv5_head,
        use_pyreader=EnvConfig.use_pyreader,
        use_random=True)
    model.build_model(image_shape)
    loss_cls, loss_bbox, rpn_cls_loss, rpn_reg_loss = model.loss()
    loss_cls.persistable = True
    loss_bbox.persistable = True
    rpn_cls_loss.persistable = True
    rpn_reg_loss.persistable = True
    loss = loss_cls + loss_bbox + rpn_cls_loss + rpn_reg_loss

    boundaries = SolverConfig.lr_steps
    gamma = SolverConfig.lr_gamma
    values = [learning_rate, learning_rate * gamma, \
      learning_rate * gamma * gamma]

    optimizer = fluid.optimizer.Momentum(
        learning_rate=exponential_with_warmup_decay(
            learning_rate=learning_rate,
            boundaries=boundaries,
            values=values,
            warmup_iter=SolverConfig.warm_up_iter,
            warmup_factor=SolverConfig.warm_up_factor),
        regularization=fluid.regularizer.L2Decay(SolverConfig.weight_decay),
        momentum=SolverConfig.momentum)
    optimizer.minimize(loss)

    fluid.memory_optimize(fluid.default_main_program())

    place = fluid.CUDAPlace(0) if EnvConfig.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    if args.pretrained_model:

        def if_exist(var):
            return os.path.exists(os.path.join(args.pretrained_model, var.name))

        fluid.io.load_vars(exe, args.pretrained_model, predicate=if_exist)

    if EnvConfig.parallel:
        train_exe = fluid.ParallelExecutor(
            use_cuda=bool(EnvConfig.use_gpu), loss_name=loss.name)

    if EnvConfig.use_pyreader:
        train_reader = reader.train(
            args,
            batch_size=TrainConfig.im_per_batch,
            total_batch_size=total_batch_size,
            padding_total=TrainConfig.padding_minibatch,
            shuffle=True)
        py_reader = model.py_reader
        py_reader.decorate_paddle_reader(train_reader)
    else:
        train_reader = reader.train(
            args, batch_size=total_batch_size, shuffle=True)
        feeder = fluid.DataFeeder(place=place, feed_list=model.feeds())

    def save_model(postfix):
        model_path = os.path.join(args.model_save_dir, postfix)
        if os.path.isdir(model_path):
            shutil.rmtree(model_path)
        fluid.io.save_persistables(exe, model_path)

    fetch_list = [loss, rpn_cls_loss, rpn_reg_loss, loss_cls, loss_bbox]

    def train_loop_pyreader():
        py_reader.start()
        smoothed_loss = SmoothedValue(args.log_window)
        try:
            start_time = time.time()
            prev_start_time = start_time
            every_pass_loss = []
            for iter_id in range(SolverConfig.max_iter):
                prev_start_time = start_time
                start_time = time.time()
                losses = train_exe.run(fetch_list=[v.name for v in fetch_list])
                every_pass_loss.append(np.mean(np.array(losses[0])))
                smoothed_loss.add_value(np.mean(np.array(losses[0])))
                lr = np.array(fluid.global_scope().find_var('learning_rate')
                              .get_tensor())
                print("Iter {:d}, lr {:.6f}, loss {:.6f}, time {:.5f}".format(
                    iter_id, lr[0],
                    smoothed_loss.get_median_value(
                    ), start_time - prev_start_time))
                rpn_cls_loss_v = np.mean(np.array(losses[1]))
                rpn_reg_loss_v = np.mean(np.array(losses[2]))
                loss_cls_v = np.mean(np.array(losses[3]))
                loss_bbox_v = np.mean(np.array(losses[4]))
                print(
                    "rpn_cls_loss: {} rpn_reg_loss: {} loss_cls: {} loss_bbox: {}".
                    format(rpn_cls_loss_v, rpn_reg_loss_v, loss_cls_v,
                           loss_bbox_v))
                sys.stdout.flush()
                if (iter_id + 1) % TrainConfig.snapshot_iter == 0:
                    save_model("model_iter{}".format(iter_id))
        except fluid.core.EOFException:
            py_reader.reset()
        return np.mean(every_pass_loss)

    def train_loop():
        start_time = time.time()
        prev_start_time = start_time
        start = start_time
        every_pass_loss = []
        smoothed_loss = SmoothedValue(args.log_window)
        for iter_id, data in enumerate(train_reader()):
            prev_start_time = start_time
            start_time = time.time()
            losses = train_exe.run(fetch_list=[v.name for v in fetch_list],
                                   feed=feeder.feed(data))
            loss_v = np.mean(np.array(losses[0]))
            every_pass_loss.append(loss_v)
            smoothed_loss.add_value(loss_v)
            lr = np.array(fluid.global_scope().find_var('learning_rate')
                          .get_tensor())
            print("Iter {:d}, lr {:.6f}, loss {:.6f}, time {:.5f}".format(
                iter_id, lr[0],
                smoothed_loss.get_median_value(), start_time - prev_start_time))
            sys.stdout.flush()
            if (iter_id + 1) % TrainConfig.snapshot_iter == 0:
                save_model("model_iter{}".format(iter_id))
            if (iter_id + 1) == SolverConfig.max_iter:
                break
        return np.mean(every_pass_loss)

    if args.use_pyreader:
        train_loop_pyreader()
    else:
        train_loop()
    save_model('model_final')


if __name__ == '__main__':
    args = parse_args()
    print_arguments(args)
    data_args = reader.Settings(args)
    train(data_args)
