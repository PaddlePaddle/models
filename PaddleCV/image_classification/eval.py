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
import logging

import paddle
import paddle.fluid as fluid
import reader
import models
from utils import *

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('data_dir',         str,  "./data/ILSVRC2012/", "The ImageNet datset")
add_arg('batch_size',       int,  256,                  "batch size on all the devices.")
add_arg('use_gpu',          bool, True,                 "Whether to use GPU or not.")
add_arg('class_dim',        int,  1000,                 "Class number.")
parser.add_argument("--pretrained_model", default=None, required=True, type=str, help="The path to load pretrained model")
add_arg('model',            str,  "ResNet50", "Set the network to use.")
add_arg('resize_short_size', int, 256,                  "Set resize short size")
add_arg('reader_thread',    int,  8,                    "The number of multi thread reader")
add_arg('reader_buf_size',  int,  2048,                 "The buf size of multi thread reader")
parser.add_argument('--image_mean', nargs='+', type=float, default=[0.485, 0.456, 0.406], help="The mean of input image data")
parser.add_argument('--image_std', nargs='+', type=float, default=[0.229, 0.224, 0.225], help="The std of input image data")
parser.add_argument('--image_shape', nargs="+",  type=int, default=[3,224,224], help=" The shape of image")
add_arg('interpolation',    int,  None,                 "The interpolation mode")
add_arg('padding_type',     str,  "SAME",               "Padding type of convolution")
add_arg('use_se',           bool, True,                 "Whether to use Squeeze-and-Excitation module for EfficientNet.")
add_arg('save_json_path',   str,  None,                 "Whether to save output in json file.")
add_arg('same_feed',        int,  0,                    "Whether to feed same images")
add_arg('print_step',       int,  1,                    "the batch step to print info")
add_arg('deploy',                bool,    False,                      "deploy mode, currently used in ACNet")
# yapf: enable

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def eval(args):
    model_list = [m for m in dir(models) if "__" not in m]
    assert args.model in model_list, "{} is not in lists: {}".format(args.model,
                                                                     model_list)
    assert os.path.isdir(
        args.pretrained_model
    ), "{} doesn't exist, please load right pretrained model path for eval".format(
        args.pretrained_model)

    assert args.image_shape[
        1] <= args.resize_short_size, "Please check the args:image_shape and args:resize_short_size, The croped size(image_shape[1]) must smaller than or equal to the resized length(resize_short_size) "

    # check gpu: when using gpu, the number of visible cards should divide batch size
    if args.use_gpu:
        assert args.batch_size % fluid.core.get_cuda_device_count(
        ) == 0, "please support correct batch_size({}), which can be divided by available cards({}), you can change the number of cards by indicating: export CUDA_VISIBLE_DEVICES= ".format(
            args.batch_size, fluid.core.get_cuda_device_count())
    image = fluid.data(
        name='image', shape=[None] + args.image_shape, dtype='float32')
    label = fluid.data(name='label', shape=[None, 1], dtype='int64')

    # model definition
    if args.model.startswith('EfficientNet'):
        model = models.__dict__[args.model](is_test=True,
                                            padding_type=args.padding_type,
                                            use_se=args.use_se)
    elif "ACNet" in args.model:
        model = models.__dict__[args.model](deploy=args.deploy)
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
    gpu_id = int(os.environ.get('FLAGS_selected_gpus', 0))

    place = fluid.CUDAPlace(gpu_id) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)

    exe.run(fluid.default_startup_program())
    if args.use_gpu:
        places = fluid.framework.cuda_places()
    else:
        places = fluid.framework.cpu_places()
    compiled_program = fluid.compiler.CompiledProgram(
        test_program).with_data_parallel(places=places)

    fluid.io.load_persistables(exe, args.pretrained_model)
    imagenet_reader = reader.ImageNetReader()
    val_reader = imagenet_reader.val(settings=args)

    # set places to run on the multi-card
    feeder = fluid.DataFeeder(place=places, feed_list=[image, label])

    test_info = [[], [], []]
    cnt = 0
    parallel_data = []
    parallel_id = []
    place_num = paddle.fluid.core.get_cuda_device_count(
    ) if args.use_gpu else int(os.environ.get('CPU_NUM', 1))
    real_iter = 0
    info_dict = {}

    for batch_id, data in enumerate(val_reader()):
        #image data and label
        image_data = [items[0:2] for items in data]
        image_id = [items[2] for items in data]
        parallel_id.append(image_id)
        parallel_data.append(image_data)
        if place_num == len(parallel_data):
            t1 = time.time()
            loss_set, acc1_set, acc5_set = exe.run(
                compiled_program,
                fetch_list=fetch_list,
                feed=list(feeder.feed_parallel(parallel_data, place_num)))
            t2 = time.time()
            period = t2 - t1
            loss = np.mean(loss_set)
            acc1 = np.mean(acc1_set)
            acc5 = np.mean(acc5_set)
            test_info[0].append(loss * len(data))
            test_info[1].append(acc1 * len(data))
            test_info[2].append(acc5 * len(data))
            cnt += len(data)
            if batch_id % args.print_step == 0:
                info = "Testbatch {0},loss {1}, acc1 {2},acc5 {3},time {4}".format(real_iter, \
                  "%.5f"%loss,"%.5f"%acc1, "%.5f"%acc5, \
                  "%2.2f sec" % period)
                logger.info(info)
                sys.stdout.flush()

            parallel_id = []
            parallel_data = []
            real_iter += 1

    test_loss = np.sum(test_info[0]) / cnt
    test_acc1 = np.sum(test_info[1]) / cnt
    test_acc5 = np.sum(test_info[2]) / cnt

    info = "Test_loss {0}, test_acc1 {1}, test_acc5 {2}".format(
        "%.5f" % test_loss, "%.5f" % test_acc1, "%.5f" % test_acc5)
    if args.save_json_path:
        info_dict = {
            "Test_loss": test_loss,
            "test_acc1": test_acc1,
            "test_acc5": test_acc5
        }
        save_json(info_dict, args.save_json_path)
    logger.info(info)
    sys.stdout.flush()


def main():
    args = parser.parse_args()
    print_arguments(args)
    check_gpu()
    check_version()
    eval(args)


if __name__ == '__main__':
    main()
