# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved
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
import shutil
import argparse
import functools
import numpy as np
import paddle.fluid as fluid
from model import network_cifar as network
import genotypes
import reader
import utility

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(utility.add_arguments, argparser=parser)

# yapf: disable
add_arg('use_multiprocess',  bool,  True,            "Whether use multiprocess reader.")
add_arg('num_workers',       int,   4,               "The multiprocess reader number.")
add_arg('data',              str,   'dataset/cifar10',"The dir of dataset.")
add_arg('batch_size',        int,   96,              "Minibatch size.")
add_arg('use_gpu',           bool,  True,            "Whether use GPU.")
add_arg('init_channels',     int,   36,              "Init channel number.")
add_arg('layers',            int,   20,              "Total number of layers.")
add_arg('class_num',         int,   10,              "Class number of dataset.")
add_arg('dropout',           float, 0.0,             "Dropout probability.")
add_arg('image_shape',       str,   "3,32,32",       "Input image size")
add_arg('model_dir',         str,   'eval_output',   "The path to load model.")
add_arg('arch',              str,   'DARTS',         "Which architecture to use")
add_arg('report_freq',       int,   10,              'Report frequency')
# yapf: enable


def build_program(main_prog, startup_prog, args):
    image_shape = [int(m) for m in args.image_shape.split(",")]
    with fluid.program_guard(main_prog, startup_prog):
        with fluid.unique_name.guard():
            image = fluid.layers.data(
                name="image", shape=image_shape, dtype="float32")
            label = fluid.layers.data(name="label", shape=[1], dtype="int64")
            genotype = eval("genotypes.%s" % args.arch)
            drop_path_prob = ''
            drop_path_mask = ''
            do_drop_path = False
            logits, logits_aux = network(
                x=image,
                is_train=False,
                c_in=args.init_channels,
                num_classes=args.class_num,
                layers=args.layers,
                auxiliary=False,
                genotype=genotype,
                do_drop_path=do_drop_path,
                drop_prob=drop_path_prob,
                drop_path_mask=drop_path_mask,
                args=args,
                name='model')
            top1 = fluid.layers.accuracy(input=logits, label=label, k=1)
            top5 = fluid.layers.accuracy(input=logits, label=label, k=5)
            loss = fluid.layers.reduce_mean(
                fluid.layers.softmax_with_cross_entropy(logits, label))
            outs = [loss, top1, top5]
    return outs


def infer(main_prog, exe, valid_reader, fetch_list, args):
    loss = utility.AvgrageMeter()
    top1 = utility.AvgrageMeter()
    top5 = utility.AvgrageMeter()
    for step_id, (image, label) in enumerate(valid_reader()):
        feed = {"image": image, "label": label}
        loss_v, top1_v, top5_v = exe.run(
            main_prog, feed=feed, fetch_list=[v.name for v in fetch_list])
        loss.update(loss_v, args.batch_size)
        top1.update(top1_v, args.batch_size)
        top5.update(top5_v, args.batch_size)
        if step_id % args.report_freq == 0:
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),\
                "Test Step {}, loss {:.6f}, acc_1 {:.6f}, acc_5 {:.6f}"\
                .format(step_id, loss.avg[0], top1.avg[0], top5.avg[0]))
    return top1.avg[0]


def main(args):
    devices = os.getenv("CUDA_VISIBLE_DEVICES") or ""
    devices_num = len(devices.split(","))

    startup_prog = fluid.Program()
    infer_prog = fluid.Program()

    infer_fetch_list = build_program(
        main_prog=infer_prog, startup_prog=startup_prog, args=args)

    infer_prog = infer_prog.clone(for_test=True)
    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(startup_prog)
    valid_reader = reader.train_valid(
        batch_size=args.batch_size, is_train=False, is_shuffle=False, args=args)
    fluid.io.load_persistables(exe, args.model_dir, main_program=infer_prog)
    infer_prog = fluid.CompiledProgram(infer_prog)

    top1 = infer(infer_prog, exe, valid_reader, infer_fetch_list, args)
    print("test_acc {:.6f}".format(top1))


if __name__ == '__main__':
    args = parser.parse_args()
    utility.print_arguments(args)
    utility.check_cuda(args.use_gpu)

    main(args)
