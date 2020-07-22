#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import os
import sys
import math
import time
import logging
import argparse
import functools
import threading
import subprocess
import numpy as np
import paddle
import paddle.fluid as fluid
import models
import reader
from losses import TripletLoss
from losses import QuadrupletLoss
from losses import EmlLoss
from losses import NpairsLoss
from utility import add_arguments, print_arguments
from utility import fmt_time, recall_topk, get_gpu_num, check_cuda

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('model', str, "ResNet50", "Set the network to use.")
add_arg('embedding_size', int, 0, "Embedding size.")
add_arg('train_batch_size', int, 120, "Minibatch size.")
add_arg('test_batch_size', int, 50, "Minibatch size.")
add_arg('image_shape', str, "3,224,224", "input image size")
add_arg('class_dim', int, 11318, "Class number.")
add_arg('lr', float, 0.0001, "set learning rate.")
add_arg('lr_strategy', str, "piecewise_decay", "Set the learning rate decay strategy.")
add_arg('lr_steps', str, "100000", "step of lr")
add_arg('total_iter_num', int, 100000, "total_iter_num")
add_arg('display_iter_step', int, 10, "display_iter_step.")
add_arg('test_iter_step', int, 5000, "test_iter_step.")
add_arg('save_iter_step', int, 5000, "save_iter_step.")
add_arg('use_gpu', bool, True, "Whether to use GPU or not.")
add_arg('pretrained_model', str, None, "Whether to use pretrained model.")
add_arg('checkpoint', str, None, "Whether to resume checkpoint.")
add_arg('model_save_dir', str, "output", "model save directory")
add_arg('loss_name', str, "triplet", "Set the loss type to use.")
add_arg('samples_each_class', int, 2, "samples_each_class.")
add_arg('margin', float, 0.1, "margin.")
add_arg('npairs_reg_lambda', float, 0.01, "npairs reg lambda.")
# yapf: enable

model_list = [m for m in dir(models) if "__" not in m]


def optimizer_setting(params):
    ls = params["learning_strategy"]
    assert ls["name"] == "piecewise_decay", \
           "learning rate strategy must be {}, but got {}".format("piecewise_decay", lr["name"])

    bd = [int(e) for e in ls["lr_steps"].split(',')]
    base_lr = params["lr"]
    lr = [base_lr * (0.1**i) for i in range(len(bd) + 1)]
    optimizer = fluid.optimizer.Momentum(
        learning_rate=fluid.layers.piecewise_decay(
            boundaries=bd, values=lr),
        momentum=0.9,
        regularization=fluid.regularizer.L2Decay(1e-4))
    return optimizer


def net_config(image, label, model, args, is_train):
    assert args.model in model_list, "{} is not in lists: {}".format(args.model,
                                                                     model_list)

    out = model.net(input=image, embedding_size=args.embedding_size)
    if not is_train:
        return None, out

    if args.loss_name == "triplet":
        metricloss = TripletLoss(margin=args.margin, )
    elif args.loss_name == "quadruplet":
        metricloss = QuadrupletLoss(
            train_batch_size=args.train_batch_size,
            samples_each_class=args.samples_each_class,
            margin=args.margin, )
    elif args.loss_name == "eml":
        metricloss = EmlLoss(
            train_batch_size=args.train_batch_size,
            samples_each_class=args.samples_each_class, )
    elif args.loss_name == "npairs":
        metricloss = NpairsLoss(
            train_batch_size=args.train_batch_size,
            samples_each_class=args.samples_each_class,
            reg_lambda=args.npairs_reg_lambda, )
    cost = metricloss.loss(out, label)
    avg_cost = fluid.layers.mean(x=cost)
    return avg_cost, out


def build_program(is_train, main_prog, startup_prog, args):
    image_shape = [int(m) for m in args.image_shape.split(",")]
    model = models.__dict__[args.model]()
    with fluid.program_guard(main_prog, startup_prog):
        queue_capacity = 64
        image = fluid.data(
                name='image', shape=[None] + image_shape, dtype='float32')
        label = fluid.data(
                name='label', shape=[None, 1], dtype='int64')
        loader = fluid.io.DataLoader.from_generator(
                feed_list=[image, label],
                capacity=queue_capacity,
                use_double_buffer=True,
                iterable=True)

        with fluid.unique_name.guard():
            avg_cost, out = net_config(image, label, model, args, is_train)
            if is_train:
                params = model.params
                params["lr"] = args.lr
                params["learning_strategy"]["lr_steps"] = args.lr_steps
                params["learning_strategy"]["name"] = args.lr_strategy
                optimizer = optimizer_setting(params)
                optimizer.minimize(avg_cost)
                global_lr = optimizer._global_learning_rate()
    """            
    if not is_train:
        main_prog = main_prog.clone(for_test=True)
    """
    if is_train:
        return loader, avg_cost, global_lr, out, label
    else:
        return loader, out


def train_async(args):
    # parameters from arguments

    logging.debug('enter train')
    model_name = args.model
    checkpoint = args.checkpoint
    pretrained_model = args.pretrained_model
    model_save_dir = args.model_save_dir

    startup_prog = fluid.Program()
    train_prog = fluid.Program()
    tmp_prog = fluid.Program()

    train_loader, train_cost, global_lr, train_feas, train_label = build_program(
        is_train=True,
        main_prog=train_prog,
        startup_prog=startup_prog,
        args=args)
    test_loader, test_feas = build_program(
        is_train=False,
        main_prog=tmp_prog,
        startup_prog=startup_prog,
        args=args)
    test_prog = tmp_prog.clone(for_test=True)

    train_fetch_list = [
        global_lr.name, train_cost.name, train_feas.name, train_label.name
    ]
    test_fetch_list = [test_feas.name]

    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    num_trainers = int(os.environ.get('PADDLE_TRAINERS_NUM', 1))
    if num_trainers <= 1 and args.use_gpu:
        places = fluid.framework.cuda_places()
    else:
        places = place

    exe.run(startup_prog)

    logging.debug('after run startup program')

    if checkpoint is not None:
        fluid.load(program=train_prog, model_path=checkpoint, executor=exe)

    if pretrained_model:

        def if_exist(var):
            return os.path.exists(os.path.join(pretrained_model, var.name))

        fluid.load(program=train_prog, model_path=pretrained_model, executor=exe)


    if args.use_gpu:
        devicenum = get_gpu_num()
    else:
        devicenum = int(os.environ.get('CPU_NUM', 1))
    assert (args.train_batch_size % devicenum) == 0
    train_batch_size = args.train_batch_size / devicenum
    test_batch_size = args.test_batch_size

    train_loader.set_sample_generator(
        reader.train(args), 
        batch_size=train_batch_size,
        drop_last=True,
        places=places)

    test_loader.set_sample_generator(
        reader.test(args),
        batch_size=test_batch_size,
        drop_last=False,
        places=place)

    train_exe = fluid.ParallelExecutor(
        main_program=train_prog,
        use_cuda=args.use_gpu,
        loss_name=train_cost.name)

    totalruntime = 0
    iter_no = 0
    train_info = [0, 0, 0]
    while iter_no <= args.total_iter_num:
        for train_batch in train_loader():
            t1 = time.time()
            lr, loss, feas, label = train_exe.run(
                                        feed=train_batch,
                                        fetch_list=train_fetch_list)
            t2 = time.time()
            period = t2 - t1
            lr = np.mean(np.array(lr))
            train_info[0] += np.mean(np.array(loss))
            train_info[1] += recall_topk(feas, label, k=1)
            train_info[2] += 1
            if iter_no % args.display_iter_step == 0:
                avgruntime = totalruntime / args.display_iter_step
                avg_loss = train_info[0] / train_info[2]
                avg_recall = train_info[1] / train_info[2]
                print("[%s] trainbatch %d, lr %.6f, loss %.6f, "\
                    "recall %.4f, time %2.2f sec" % \
                    (fmt_time(), iter_no, lr, avg_loss, avg_recall, avgruntime))
                sys.stdout.flush()
                totalruntime = 0
            if iter_no % 1000 == 0:
                train_info = [0, 0, 0]

            totalruntime += period

            if iter_no % args.test_iter_step == 0 and iter_no != 0:
                f, l = [], []
                for batch_id, test_batch in enumerate(test_loader()):
                    t1 = time.time()
                    [feas] = exe.run(test_prog,
                                 feed=test_batch,
                                 fetch_list=test_fetch_list)

                    label = np.asarray(test_batch[0]['label'])
                    label = np.squeeze(label)
                    f.append(feas)
                    l.append(label)

                    t2 = time.time()
                    period = t2 - t1
                    if batch_id % 20 == 0:
                        print("[%s] testbatch %d, time %2.2f sec" % \
                            (fmt_time(), batch_id, period))

                f = np.vstack(f)
                l = np.hstack(l)
                recall = recall_topk(f, l, k=1)
                print("[%s] test_img_num %d, trainbatch %d, test_recall %.5f" % \
                    (fmt_time(), len(f), iter_no, recall))
                sys.stdout.flush()

            if iter_no % args.save_iter_step == 0 and iter_no != 0:
                model_path = os.path.join(model_save_dir + '/' + model_name,
                                      str(iter_no))
                if not os.path.isdir(model_path):
                    os.makedirs(model_path)
                fluid.save(program=train_prog, model_path=model_path)

            iter_no += 1


def initlogging():
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    loglevel = logging.DEBUG
    logging.basicConfig(
        level=loglevel,
        # logger.BASIC_FORMAT,
        format="%(levelname)s:%(filename)s[%(lineno)s] %(name)s:%(funcName)s->%(message)s",
        datefmt='%a, %d %b %Y %H:%M:%S')


def main():
    args = parser.parse_args()
    print_arguments(args)
    check_cuda(args.use_gpu)
    train_async(args)


if __name__ == '__main__':
    main()
