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
parser.add_argument("--pretrained_model", default=None, required=True, type=str, help="The path to load pretrained model")
add_arg('model',            str,  "ResNet50",           "Set the network to use.")
add_arg('save_inference',   bool, False,                "Whether to save inference model or not")
add_arg('resize_short_size',int,  256,                  "Set resize short size")
add_arg('reader_thread',    int,  1,                    "The number of multi thread reader")
add_arg('reader_buf_size',  int,  2048,                 "The buf size of multi thread reader")
parser.add_argument('--image_mean', nargs='+', type=float, default=[0.485, 0.456, 0.406], help="The mean of input image data")
parser.add_argument('--image_std', nargs='+', type=float, default=[0.229, 0.224, 0.225], help="The std of input image data")
parser.add_argument('--image_shape', nargs='+', type=int, default=[3, 224, 224], help="the shape of image")
add_arg('topk',             int,  1,                    "topk")
add_arg('class_map_path',   str,  "./utils/tools/readable_label.txt", "readable label filepath")
add_arg('interpolation',    int,  None,                 "The interpolation mode")
add_arg('padding_type',     str,  "SAME",               "Padding type of convolution")
add_arg('use_se',           bool, True,                 "Whether to use Squeeze-and-Excitation module for EfficientNet.")
add_arg('image_path',       str,  None,                 "single image path")
add_arg('batch_size',       int,  8,                    "batch_size on all devices")
add_arg('save_json_path',        str,  None,            "save output to a json file")
# yapf: enable


def infer(args):
    model_list = [m for m in dir(models) if "__" not in m]
    assert args.model in model_list, "{} is not in lists: {}".format(args.model,
                                                                     model_list)
    assert os.path.isdir(args.pretrained_model
                         ), "please load right pretrained model path for infer"

    assert args.image_shape[
        1] <= args.resize_short_size, "Please check the args:image_shape and args:resize_short_size, The croped size(image_shape[1]) must smaller than or equal to the resized length(resize_short_size) "

    if args.image_path:
        assert os.path.isfile(
            args.image_path
        ), "Please check the args:image_path, it should be a path to single image."
        if args.use_gpu:
            assert fluid.core.get_cuda_device_count(
            ) == 1, "please set \"export CUDA_VISIBLE_DEVICES=\" available single card"
        else:
            assert int(os.environ.get('CPU_NUM',
                                      1)) == 1, "please set CPU_NUM as 1"

    image = fluid.data(
        name='image', shape=[None] + args.image_shape, dtype='float32')

    if args.model.startswith('EfficientNet'):
        model = models.__dict__[args.model](is_test=True,
                                            padding_type=args.padding_type,
                                            use_se=args.use_se)
    else:
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

    compiled_program = fluid.compiler.CompiledProgram(
        test_program).with_data_parallel()

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

    imagenet_reader = reader.ImageNetReader()
    test_reader = imagenet_reader.test(settings=args)

    feeder = fluid.DataFeeder(place=place, feed_list=[image])
    test_reader = feeder.decorate_reader(test_reader, multi_devices=True)

    TOPK = args.topk
    if os.path.exists(args.class_map_path):
        print("The map of readable label and numerical label has been found!")
        f = open(args.class_map_path)
        label_dict = {}
        for item in f.readlines():
            key = item.split(" ")[0]
            value = [l.replace("\n", "") for l in item.split(" ")[1:]]
            label_dict[key] = value

    for batch_id, data in enumerate(test_reader()):
        result = exe.run(compiled_program, fetch_list=fetch_list, feed=data)
        result = result[0][0]
        pred_label = np.argsort(result)[::-1][:TOPK]

        if os.path.exists(args.class_map_path):
            readable_pred_label = []
            for label in pred_label:
                readable_pred_label.append(label_dict[str(label)])
                print(readable_pred_label)
            info = "Test-{0}-score: {1}, class{2} {3}".format(
                batch_id, result[pred_label], pred_label, readable_pred_label)
        else:
            info = "Test-{0}-score: {1}, class{2}".format(
                batch_id, result[pred_label], pred_label)
        print(info)
        if args.save_json_path:
            save_json(info, args.save_json_path)

        sys.stdout.flush()
    if args.image_path:
        os.remove(".tmp.txt")


def main():
    args = parser.parse_args()
    print_arguments(args)
    check_gpu()
    check_version()
    infer(args)


if __name__ == '__main__':
    main()
