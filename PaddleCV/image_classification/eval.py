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
import time
import sys
import math
import numpy as np
import argparse
import functools

import paddle
import paddle.fluid as fluid
import reader
import models
from utils import *

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('data_dir',         str,  "./data/ILSVRC2012/", "The ImageNet datset")
add_arg('batch_size',       int,  256,                  "Minibatch size.")
add_arg('use_gpu',          bool, True,                 "Whether to use GPU or not.")
add_arg('class_dim',        int,  1000,                 "Class number.")
add_arg('image_shape',      str,  "3,224,224",          "Input image size")
parser.add_argument("--pretrained_model", default=None, required=True, type=str, help="The path to load pretrained model")
add_arg('model',            str,  "ResNet50", "Set the network to use.")
add_arg('resize_short_size', int, 256,                  "Set resize short size")
add_arg('reader_thread',    int,  8,                    "The number of multi thread reader")
add_arg('reader_buf_size',  int,  2048,                 "The buf size of multi thread reader")
parser.add_argument('--image_mean', nargs='+', type=float, default=[0.485, 0.456, 0.406], help="The mean of input image data")
parser.add_argument('--image_std', nargs='+', type=float, default=[0.229, 0.224, 0.225], help="The std of input image data")
add_arg('crop_size',        int,  224,                  "The value of crop size")
add_arg('interpolation',    int,  None,                 "The interpolation mode")
add_arg('padding_type',     str,  "SAME",               "Padding type of convolution")
# yapf: enable


def eval(args):
    image_shape = [int(m) for m in args.image_shape.split(",")]

    model_list = [m for m in dir(models) if "__" not in m]
    assert args.model in model_list, "{} is not in lists: {}".format(args.model,
                                                                     model_list)
    assert os.path.isdir(
        args.pretrained_model
    ), "{} doesn't exist, please load right pretrained model path for eval".format(
        args.pretrained_model)

    image = fluid.data(
        name='image', shape=[None] + image_shape, dtype='float32')
    label = fluid.data(name='label', shape=[None, 1], dtype='int64')

    # model definition
    if args.model.startswith('EfficientNet'):
        model = models.__dict__[args.model](is_test=True,
                                            padding_type=args.padding_type)
    else:
        model = models.__dict__[args.model]()

    if args.model == "GoogLeNet":
        out0, out1, out2 = model.net(input=image, class_dim=args.class_dim)
        cost0 = fluid.layers.cross_entropy(input=out0, label=label)
        cost1 = fluid.layers.cross_entropy(input=out1, label=label)
        cost2 = fluid.layers.cross_entropy(input=out2, label=label)
        avg_cost0 = fluid.layers.mean(x=cost0)
        avg_cost1 = fluid.layers.mean(x=cost1)
        avg_cost2 = fluid.layers.mean(x=cost2)

        avg_cost = avg_cost0 + 0.3 * avg_cost1 + 0.3 * avg_cost2
        acc_top1 = fluid.layers.accuracy(input=out0, label=label, k=1)
        acc_top5 = fluid.layers.accuracy(input=out0, label=label, k=5)
    else:
        out = model.net(input=image, class_dim=args.class_dim)

        cost, pred = fluid.layers.softmax_with_cross_entropy(
            out, label, return_softmax=True)
        avg_cost = fluid.layers.mean(x=cost)
        acc_top1 = fluid.layers.accuracy(input=pred, label=label, k=1)
        acc_top5 = fluid.layers.accuracy(input=pred, label=label, k=5)

    test_program = fluid.default_main_program().clone(for_test=True)

    fetch_list = [avg_cost.name, acc_top1.name, acc_top5.name]

    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    fluid.io.load_persistables(exe, args.pretrained_model)
    imagenet_reader = reader.ImageNetReader()
    val_reader = imagenet_reader.val(settings=args)

    feeder = fluid.DataFeeder(place=place, feed_list=[image, label])

    test_info = [[], [], []]
    cnt = 0
    for batch_id, data in enumerate(val_reader()):
        t1 = time.time()
        loss, acc1, acc5 = exe.run(test_program,
                                   fetch_list=fetch_list,
                                   feed=feeder.feed(data))
        t2 = time.time()
        period = t2 - t1
        loss = np.mean(loss)
        acc1 = np.mean(acc1)
        acc5 = np.mean(acc5)
        test_info[0].append(loss * len(data))
        test_info[1].append(acc1 * len(data))
        test_info[2].append(acc5 * len(data))
        cnt += len(data)
        if batch_id % 10 == 0:
            print("Testbatch {0},loss {1}, "
                  "acc1 {2},acc5 {3},time {4}".format(batch_id, \
                  "%.5f"%loss,"%.5f"%acc1, "%.5f"%acc5, \
                  "%2.2f sec" % period))
            sys.stdout.flush()

    test_loss = np.sum(test_info[0]) / cnt
    test_acc1 = np.sum(test_info[1]) / cnt
    test_acc5 = np.sum(test_info[2]) / cnt

    print("Test_loss {0}, test_acc1 {1}, test_acc5 {2}".format(
        "%.5f" % test_loss, "%.5f" % test_acc1, "%.5f" % test_acc5))
    sys.stdout.flush()


def main():
    args = parser.parse_args()
    print_arguments(args)
    check_gpu()
    check_version()
    eval(args)


if __name__ == '__main__':
    main()
