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
sys.path.append('/cv/workspace/PaddleSlim')
from paddleslim.quant import quant_aware, convert
parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('data_dir',         str,  "./data/ILSVRC2012/", "The ImageNet datset")
add_arg('batch_size',       int,  200,                  "Minibatch size.")
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
add_arg('model_name', str,  None, "model filename for inference model")
add_arg('params_name', str, None, "params filename for inference model")

# yapf: enable


def eval(args):
    image_shape = args.image_shape

    model_list = [m for m in dir(models) if "__" not in m]
    assert args.model in model_list, "{} is not in lists: {}".format(args.model,
                                                                     model_list)
    assert os.path.isdir(
        args.pretrained_model
    ), "{} doesn't exist, please load right pretrained model path for eval".format(
        args.pretrained_model)

    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()

    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    infer_prog, feed_names, fetch_list = fluid.io.load_inference_model(
        dirname=args.pretrained_model,
        executor=exe,
        model_filename=args.model_name,
        params_filename=args.params_name)

    imagenet_reader = reader.ImageNetReader()
    val_reader = imagenet_reader.val(settings=args)

    feeder = fluid.DataFeeder(
        place=place, feed_list=feed_names, program=infer_prog)

    test_info = [[], []]
    cnt = 0
    for batch_id, data in enumerate(val_reader()):
        t1 = time.time()
        image = [[d[0]] for d in data]
        label = [[d[1]] for d in data]
        feed_data = feeder.feed(image)
        pred = exe.run(infer_prog, fetch_list=fetch_list, feed=feed_data)
        t2 = time.time()
        period = t2 - t1
        pred = np.array(pred[0])
        label = np.array(label)
        sort_array = pred.argsort(axis=1)
        top_1_pred = sort_array[:, -1:][:, ::-1]
        top_1 = np.mean(label == top_1_pred)
        top_5_pred = sort_array[:, -5:][:, ::-1]
        acc_num = 0
        for i in range(len(label)):
            if label[i][0] in top_5_pred[i]:
                acc_num += 1
        test_info[0].append(top_1 * len(data))
        test_info[1].append(acc_num)
        cnt += len(data)

        if batch_id % 10 == 0:
            print("Testbatch {0}, "
                  "acc1 {1},acc5 {2},time {3}".format(batch_id, \
                  "%.5f"%top_1, "%.5f"%(acc_num/len(data)), \
                  "%2.2f sec" % period))
            sys.stdout.flush()

    test_acc1 = np.sum(test_info[0]) / cnt
    test_acc5 = np.sum(test_info[1]) / cnt

    print("total image {0} test_acc1 {1}, test_acc5 {2}".format(
        cnt, "%.5f" % test_acc1, "%.5f" % test_acc5))
    sys.stdout.flush()


def main():
    args = parser.parse_args()
    print_arguments(args)
    check_gpu()
    check_version()
    eval(args)


if __name__ == '__main__':
    main()
