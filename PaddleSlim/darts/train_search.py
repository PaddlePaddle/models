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
np.set_printoptions(formatter={'float': '{: 0.4f}'.format})


def set_paddle_flags(flags):
    for key, value in flags.items():
        if os.environ.get(key, None) is None:
            os.environ[key] = str(value)


# NOTE(paddle-dev): All of these flags should be
# set before `import paddle`. Otherwise, it would
# not take any effect. 
set_paddle_flags({
    'FLAGS_eager_delete_tensor_gb': 0,  # enable GC 
    # You can omit the following settings, because the default
    # value of FLAGS_memory_fraction_of_eager_deletion is 1,
    # and default value of FLAGS_fast_eager_deletion_mode is 1 
    'FLAGS_memory_fraction_of_eager_deletion': 1,
    'FLAGS_fast_eager_deletion_mode': 1,
    # Setting the default used gpu memory
    'FLAGS_fraction_of_gpu_memory_to_use': 0.92
})

import paddle.fluid as fluid
import paddle.fluid.profiler as profiler
import reader
import utility
import architect
from model_search import model, get_genotype

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(utility.add_arguments, argparser=parser)

# yapf: disable
add_arg('profile',           bool,  False,           "Enable profiler.")
add_arg('report_freq',       int,   50,              "Report frequency.")
add_arg('use_multiprocess',  bool,  True,            "Whether use multiprocess reader.")
add_arg('num_workers',       int,   8,               "The multiprocess reader number.")
add_arg('data',              str,   'dataset/cifar10',"The dir of dataset.")
add_arg('batch_size',        int,   16,              "Minibatch size.")
add_arg('learning_rate',     float, 0.025,           "The start learning rate.")
add_arg('learning_rate_min', float, 0.001,           "The min learning rate.")
add_arg('momentum',          float, 0.9,             "Momentum.")
add_arg('weight_decay',      float, 3e-4,            "Weight_decay.")
add_arg('use_gpu',           bool,  True,            "Whether use GPU.")
add_arg('epochs',            int,   50,              "Epoch number.")
add_arg('init_channels',     int,   16,              "Init channel number.")
add_arg('layers',            int,   8,               "Total number of layers.")
add_arg('class_num',         int,   10,              "Class number of dataset.")
add_arg('model_save_dir',    str,   'search_output',  "The path to save model.")
add_arg('cutout',            bool,  True,            'Whether use cutout.')
add_arg('cutout_length',     int,   16,              "Cutout length.")
add_arg('grad_clip',         float, 5,               "Gradient clipping.")
add_arg('train_portion',     float, 0.5,             "Portion of training data.")
add_arg('arch_learning_rate',float, 3e-4,            "Learning rate for arch encoding.")
add_arg('arch_weight_decay', float, 1e-3,            "Weight decay for arch encoding.")
add_arg('image_shape',       str,   "3,32,32",       "input image size")
add_arg('with_mem_opt',      bool,  False,           "Whether to use memory optimization or not.")
#parser.add_argument('--unrolled', action='store_true', help='If set, use one-step unrolled validation loss.')
# yapf: enable

CIFAR10 = 50000


def genotype(test_prog, exe, place):
    image_shape = [1] + [int(m) for m in args.image_shape.split(",")]
    label_shape = [1, 1]
    image_tensor = fluid.LoDTensor()
    image_tensor.set(np.random.random(size=image_shape).astype('float32'),
                     place)
    label_tensor = fluid.LoDTensor()
    label_tensor.set(np.random.random(size=label_shape).astype('int64'), place)
    arch_names = utility.get_parameters(
        test_prog.global_block().all_parameters(), 'arch')[0]
    feed = {
        "image_train": image_tensor,
        "label_train": label_tensor,
        "image_val": image_tensor,
        "label_val": label_tensor
    }
    arch_values = exe.run(test_prog, feed=feed, fetch_list=arch_names)
    # softmax
    arch_values = [
        np.exp(arch_v) / np.sum(np.exp(arch_v)) for arch_v in arch_values
    ]
    alpha_normal = [
        i for i in zip(arch_names, arch_values) if 'weight1' in i[0]
    ]
    alpha_reduce = [
        i for i in zip(arch_names, arch_values) if 'weight2' in i[0]
    ]
    print('normal:')
    print(
        np.array([
            pair[1]
            for pair in sorted(
                alpha_normal, key=lambda i: int(i[0].split('_')[1]))
        ]))
    print('reduce:')
    print(
        np.array([
            pair[1]
            for pair in sorted(
                alpha_reduce, key=lambda i: int(i[0].split('_')[1]))
        ]))
    genotype = get_genotype(arch_names, arch_values)
    print("genotype={}".format(genotype))


def valid(epoch_id, valid_reader, fetch_list, test_prog, exe):
    loss = utility.AvgrageMeter()
    top1 = utility.AvgrageMeter()
    top5 = utility.AvgrageMeter()
    for step_id, (image_val, label_val) in enumerate(valid_reader()):
        # use valid data to feed image_train and label_train
        feed = {
            "image_train": image_val,
            "label_train": label_val,
            "image_val": image_val,
            "label_val": label_val
        }
        loss_v, top1_v, top5_v = exe.run(test_prog,
                                         feed=feed,
                                         fetch_list=fetch_list)
        loss.update(loss_v, args.batch_size)
        top1.update(top1_v, args.batch_size)
        top5.update(top5_v, args.batch_size)
        if step_id % args.report_freq == 0:
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), \
                "Valid Epoch {}, Step {}, loss {:.3f}, acc_1 {:.6f}, acc_5 {:.6f}"\
                .format(epoch_id, step_id, loss.avg[0], top1.avg[0], top5.avg[0]))
    return top1.avg[0]


def train(epoch_id, train_reader, valid_reader, fetch_list, arch_progs_list,
          train_prog, exe):
    loss = utility.AvgrageMeter()
    top1 = utility.AvgrageMeter()
    top5 = utility.AvgrageMeter()
    for step_id, ((image_train, label_train), (
            image_val,
            label_val)) in enumerate(zip(train_reader(), valid_reader())):
        if args.profile:
            if epoch_id == 0 and step_id == 1:
                profiler.start_profiler("All")
            elif epoch_id == 0 and step_id == 3:
                profiler.stop_profiler("total", "/tmp/profile")
        feed = {
            "image_train": image_train,
            "label_train": label_train,
            "image_val": image_val,
            "label_val": label_val
        }
        exe.run(arch_progs_list[0], feed=feed)
        exe.run(arch_progs_list[1], feed=feed)
        lr, loss_v, top1_v, top5_v = exe.run(
            train_prog, feed=feed, fetch_list=[v.name for v in fetch_list])
        loss.update(loss_v, args.batch_size)
        top1.update(top1_v, args.batch_size)
        top5.update(top5_v, args.batch_size)
        if step_id % args.report_freq == 0:
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), \
                "Train Epoch {}, Step {}, Lr {:.8f}, loss {:.6f}, acc_1 {:.6f}, acc_5 {:.6f}"\
                .format(epoch_id, step_id, lr[0], loss.avg[0], top1.avg[0], top5.avg[0]))
    return top1.avg[0]


def main(args):
    devices = os.getenv("CUDA_VISIBLE_DEVICES") or ""
    devices_num = len(devices.split(","))
    step_per_epoch = int(CIFAR10 * args.train_portion / args.batch_size)
    is_shuffle = True

    startup_prog = fluid.Program()
    data_prog = fluid.Program()
    test_prog = fluid.Program()

    image_shape = [int(m) for m in args.image_shape.split(",")]
    print(
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "Constructing graph...")
    with fluid.unique_name.guard():
        with fluid.program_guard(data_prog, startup_prog):
            image_train = fluid.layers.data(
                name="image_train", shape=image_shape, dtype="float32")
            label_train = fluid.layers.data(
                name="label_train", shape=[1], dtype="int64")
            image_val = fluid.layers.data(
                name="image_val", shape=image_shape, dtype="float32")
            label_val = fluid.layers.data(
                name="label_val", shape=[1], dtype="int64")
            learning_rate = fluid.layers.cosine_decay(
                args.learning_rate, 4 * step_per_epoch, args.epochs)
            # Pytorch CosineAnnealingLR
            learning_rate = learning_rate / args.learning_rate * (
                args.learning_rate - args.learning_rate_min
            ) + args.learning_rate_min

        arch_progs_list, fetch = architect.compute_unrolled_step(
            image_train, label_train, image_val, label_val, data_prog,
            startup_prog, learning_rate, args)

        train_prog = data_prog.clone()
        with fluid.program_guard(train_prog, startup_prog):
            logits, loss = model(
                image_train,
                label_train,
                args.init_channels,
                args.class_num,
                args.layers,
                name="model")
            top1 = fluid.layers.accuracy(input=logits, label=label_train, k=1)
            top5 = fluid.layers.accuracy(input=logits, label=label_train, k=5)
            print("param size = {:.6f}MB".format(
                utility.count_parameters_in_MB(train_prog.global_block()
                                               .all_parameters(), 'model')))
            test_prog = train_prog.clone(for_test=True)

            model_var = utility.get_parameters(
                train_prog.global_block().all_parameters(), 'model')[1]
            # update model_var with gradientclip
            fluid.clip.set_gradient_clip(
                clip=fluid.clip.GradientClipByGlobalNorm(clip_norm=5.0),
                param_list=[v.name for v in model_var])
            follower_opt = fluid.optimizer.MomentumOptimizer(
                learning_rate,
                args.momentum,
                regularization=fluid.regularizer.L2DecayRegularizer(
                    args.weight_decay))
            follower_opt.minimize(
                loss, parameter_list=[v.name for v in model_var])

    print(
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "Construct graph done")
    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(startup_prog)
    train_reader, valid_reader = reader.train_search(
        batch_size=args.batch_size,
        train_portion=args.train_portion,
        is_shuffle=is_shuffle,
        args=args)

    exec_strategy = fluid.ExecutionStrategy()
    exec_strategy.num_threads = 4 * devices_num
    build_strategy = fluid.BuildStrategy()
    if args.with_mem_opt:
        learning_rate.persistable = True
        loss.persistable = True
        top1.persistable = True
        top5.persistable = True
        build_strategy.enable_inplace = True
        build_strategy.memory_optimize = True
    arch_progs_list[0] = fluid.CompiledProgram(arch_progs_list[
        0]).with_data_parallel(
            loss_name=fetch[0].name,
            build_strategy=build_strategy,
            exec_strategy=exec_strategy)
    arch_progs_list[1] = fluid.CompiledProgram(arch_progs_list[
        1]).with_data_parallel(
            loss_name=fetch[1].name,
            build_strategy=build_strategy,
            exec_strategy=exec_strategy)
    parallel_train_prog = fluid.CompiledProgram(train_prog).with_data_parallel(
        loss_name=loss.name,
        build_strategy=build_strategy,
        exec_strategy=exec_strategy)
    compiled_test_prog = fluid.CompiledProgram(test_prog)

    def save_model(postfix, program):
        model_path = os.path.join(args.model_save_dir, postfix)
        if os.path.isdir(model_path):
            shutil.rmtree(model_path)
        print('save models to %s' % (model_path))
        fluid.io.save_persistables(exe, model_path, main_program=program)

    for epoch_id in range(args.epochs):
        # get genotype
        genotype(test_prog, exe, place)
        train_fetch_list = [learning_rate, loss, top1, top5]
        train_top1 = train(epoch_id, train_reader, valid_reader,
                           train_fetch_list, arch_progs_list,
                           parallel_train_prog, exe)
        print("Epoch {}, train_acc {:.6f}".format(epoch_id, train_top1))
        valid_fetch_list = [loss, top1, top5]
        valid_top1 = valid(epoch_id, valid_reader, valid_fetch_list,
                           compiled_test_prog, exe)
        print("Epoch {}, valid_acc {:.6f}".format(epoch_id, valid_top1))
        save_model('search_' + str(epoch_id), train_prog)


if __name__ == '__main__':
    args = parser.parse_args()
    utility.print_arguments(args)
    utility.check_cuda(args.use_gpu)

    main(args)
