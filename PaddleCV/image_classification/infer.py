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
# yapf: disable
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('data_dir',         str,  "./data/ILSVRC2012/", "The ImageNet data")
add_arg('use_gpu',          bool, True,                 "Whether to use GPU or not.")
add_arg('class_dim',        int,  1000,                 "Class number.")
add_arg('image_shape',      str,  "3,224,224",          "Input image size")
parser.add_argument("--pretrained_model", default=None, required=True, type=str, help="The path to load pretrained model")
add_arg('model',            str,  "ResNet50",            "Set the network to use.")
add_arg('save_inference',   bool, False,                "Whether to save inference model or not")
add_arg('resize_short_size',int,  256,                  "Set resize short size")
add_arg('reader_thread',    int,  1,                    "The number of multi thread reader")
add_arg('reader_buf_size',  int,  2048,                 "The buf size of multi thread reader")
parser.add_argument('--image_mean', nargs='+', type=float, default=[0.485, 0.456, 0.406], help="The mean of input image data")
parser.add_argument('--image_std', nargs='+', type=float, default=[0.229, 0.224, 0.225], help="The std of input image data")
add_arg('crop_size',        int,  224,                  "The value of crop size")
add_arg('topk',             int,  1,                    "topk")
add_arg('label_path',       str,  "./utils/tools/readable_label.txt", "readable label filepath")
add_arg('interpolation',    int,  None,                 "The interpolation mode")
# yapf: enable


def infer(args):
    image_shape = [int(m) for m in args.image_shape.split(",")]
    model_list = [m for m in dir(models) if "__" not in m]
    assert args.model in model_list, "{} is not in lists: {}".format(args.model,
                                                                     model_list)
    assert os.path.isdir(args.pretrained_model
                         ), "please load right pretrained model path for infer"
    image = fluid.data(
        name='image', shape=[None] + image_shape, dtype='float32')
    model = models.__dict__[args.model]()
    if args.model == "GoogLeNet":
        out, _, _ = model.net(input=image, class_dim=args.class_dim)
    else:
        out = model.net(input=image, class_dim=args.class_dim)
        out = fluid.layers.softmax(out)

    test_program = fluid.default_main_program().clone(for_test=True)

    fetch_list = [out.name]

    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    fluid.io.load_persistables(exe, args.pretrained_model)
    if args.save_inference:
        fluid.io.save_inference_model(
            dirname=args.model,
            feeded_var_names=['image'],
            main_program=test_program,
            target_vars=out,
            executor=exe,
            model_filename='model',
            params_filename='params')
        print("model: ", args.model, " is already saved")
        exit(0)

    args.test_batch_size = 1
    imagenet_reader = reader.ImageNetReader()
    test_reader = imagenet_reader.test(settings=args)
    feeder = fluid.DataFeeder(place=place, feed_list=[image])

    TOPK = args.topk
    assert os.path.exists(args.label_path), "Index file doesn't exist!"
    f = open(args.label_path)
    label_dict = {}
    for item in f.readlines():
        key = item.split(" ")[0]
        value = [l.replace("\n", "") for l in item.split(" ")[1:]]
        label_dict[key] = value

    for batch_id, data in enumerate(test_reader()):
        result = exe.run(test_program,
                         fetch_list=fetch_list,
                         feed=feeder.feed(data))
        result = result[0][0]
        pred_label = np.argsort(result)[::-1][:TOPK]
        readable_pred_label = []
        for label in pred_label:
            readable_pred_label.append(label_dict[str(label)])
        print("Test-{0}-score: {1}, class{2} {3}".format(batch_id, result[
            pred_label], pred_label, readable_pred_label))
        sys.stdout.flush()


def main():
    args = parser.parse_args()
    print_arguments(args)
    check_gpu()
    check_version()
    infer(args)


if __name__ == '__main__':
    main()
