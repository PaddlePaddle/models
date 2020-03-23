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
"""Trainer for OCR CTC or attention model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import paddle.fluid as fluid
from utility import add_arguments, print_arguments, to_lodtensor, get_ctc_feeder_data, get_attention_feeder_data
from utility import check_gpu
import paddle.fluid.profiler as profiler
from crnn_ctc_model import ctc_train_net
from attention_model import attention_train_net
import data_reader
import argparse
import functools
import sys
import time
import os
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('batch_size',        int,     32,         "Minibatch size.")

add_arg('lr',                float,   1e-3,         "Learning rate.")
add_arg('lr_decay_strategy', str,     None,         "Learning rate decay strategy. 'piecewise_decay' or None is valid.")
add_arg('l2decay',           float,   4e-4,         "L2 decay rate.")
add_arg('momentum',          float,   0.9,         "Momentum rate.")
add_arg('gradient_clip',     float,   10.0,         "The threshold of gradient clipping.")

add_arg('total_step',        int,     720000,    "The number of iterations. Zero or less means whole training set. More than 0 means the training set might be looped until # of iterations is reached.")
add_arg('log_period',        int,     1000,       "Log period.")
add_arg('save_model_period', int,     15000,      "Save model period. '-1' means never saving the model.")
add_arg('eval_period',       int,     15000,      "Evaluate period. '-1' means never evaluating the model.")
add_arg('save_model_dir',    str,     "./models", "The directory the model to be saved to.")
add_arg('train_images',      str,     None,       "The directory of images to be used for training.")
add_arg('train_list',        str,     None,       "The list file of images to be used for training.")
add_arg('test_images',       str,     None,       "The directory of images to be used for test.")
add_arg('test_list',         str,     None,       "The list file of images to be used for training.")
add_arg('model',             str,     "crnn_ctc",           "Which type of network to be used. 'crnn_ctc' or 'attention'")
add_arg('init_model',        str,     None,       "The init model file of directory.")
add_arg('use_gpu',           bool,    True,      "Whether use GPU to train.")
add_arg('min_average_window',int,     10000,     "Min average window.")
add_arg('max_average_window',int,     12500,     "Max average window. It is proposed to be set as the number of minibatch in a pass.")
add_arg('average_window',    float,   0.15,      "Average window.")
add_arg('parallel',          bool,    False,     "Whether use parallel training.")
add_arg('profile',           bool,    False,      "Whether to use profiling.")
add_arg('skip_batch_num',    int,     0,          "The number of first minibatches to skip as warm-up for better performance test.")
add_arg('skip_test',         bool,    False,      "Whether to skip test phase.")
# yapf: enable


def train(args):
    """OCR training"""

    if args.model == "crnn_ctc":
        train_net = ctc_train_net
        get_feeder_data = get_ctc_feeder_data
    else:
        train_net = attention_train_net
        get_feeder_data = get_attention_feeder_data

    num_classes = None
    num_classes = data_reader.num_classes(
    ) if num_classes is None else num_classes
    data_shape = data_reader.data_shape()
    # define network
    sum_cost, error_evaluator, inference_program, model_average = train_net(
        args, data_shape, num_classes)

    # data reader
    train_reader = data_reader.train(
        args.batch_size,
        train_images_dir=args.train_images,
        train_list_file=args.train_list,
        cycle=args.total_step > 0,
        model=args.model)
    test_reader = data_reader.test(
        test_images_dir=args.test_images,
        test_list_file=args.test_list,
        model=args.model)

    # prepare environment
    place = fluid.CPUPlace()
    if args.use_gpu:
        place = fluid.CUDAPlace(0)
    exe = fluid.Executor(place)

    if 'ce_mode' in os.environ:
        fluid.default_startup_program().random_seed = 90

    exe.run(fluid.default_startup_program())

    # load init model
    if args.init_model is not None:
        model_dir = args.init_model
        fluid.load(
            fluid.default_main_program(),
            model_dir,
            var_list=fluid.io.get_program_parameter(fluid.default_main_program(
            )))
        print("Init model from: %s." % args.init_model)

    train_exe = exe
    error_evaluator.reset(exe)
    if args.parallel:
        train_exe = fluid.ParallelExecutor(
            use_cuda=True if args.use_gpu else False, loss_name=sum_cost.name)

    fetch_vars = [sum_cost] + error_evaluator.metrics

    def train_one_batch(data):
        var_names = [var.name for var in fetch_vars]
        if args.parallel:
            results = train_exe.run(var_names,
                                    feed=get_feeder_data(data, place))
            results = [np.array(result).sum() for result in results]
        else:
            results = train_exe.run(feed=get_feeder_data(data, place),
                                    fetch_list=fetch_vars)
            results = [result[0] for result in results]
        return results

    def test(iter_num):
        error_evaluator.reset(exe)
        for data in test_reader():
            exe.run(inference_program, feed=get_feeder_data(data, place))
        _, test_seq_error = error_evaluator.eval(exe)
        print("\n[%s] - Iter[%d]; Test seq error: %s.\n" %
              (time.asctime(time.localtime(time.time())), iter_num,
               str(test_seq_error[0])))

        #Note: The following logs are special for CE monitoring.
        #Other situations do not need to care about these logs.
        if 'ce_mode' in os.environ:
            print("kpis	test_acc	%f" % (1 - test_seq_error[0]))

    def save_model(args, exe, iter_num):
        filename = "model_%05d" % iter_num
        fluid.save(fluid.default_main_program(),
                   os.path.join(args.save_model_dir, filename))
        print("Saved model to: %s/%s." % (args.save_model_dir, filename))

    iter_num = 0
    stop = False
    start_time = time.time()
    while not stop:
        total_loss = 0.0
        total_seq_error = 0.0
        batch_times = []
        # train a pass
        for data in train_reader():
            if args.total_step > 0 and iter_num == args.total_step + args.skip_batch_num:
                stop = True
                break
            if iter_num < args.skip_batch_num:
                print("Warm-up iteration")
            if iter_num == args.skip_batch_num:
                profiler.reset_profiler()
            start = time.time()
            results = train_one_batch(data)
            batch_time = time.time() - start
            fps = args.batch_size / batch_time
            batch_times.append(batch_time)
            total_loss += results[0]
            total_seq_error += results[2]

            iter_num += 1
            # training log
            if iter_num % args.log_period == 0:
                print("\n[%s] - Iter[%d]; Avg loss: %.3f; Avg seq err: %.3f" %
                      (time.asctime(time.localtime(time.time())), iter_num,
                       total_loss / (args.log_period * args.batch_size),
                       total_seq_error / (args.log_period * args.batch_size)))
                if 'ce_mode' in os.environ:
                    print("kpis	train_cost	%f" %
                          (total_loss / (args.log_period * args.batch_size)))
                    print("kpis	train_acc	%f" %
                          (1 - total_seq_error /
                           (args.log_period * args.batch_size)))
                total_loss = 0.0
                total_seq_error = 0.0

            # evaluate
            if not args.skip_test and iter_num % args.eval_period == 0:
                if model_average:
                    with model_average.apply(exe):
                        test(iter_num)
                else:
                    test(iter_num)

            # save model
            if iter_num % args.save_model_period == 0:
                if model_average:
                    with model_average.apply(exe):
                        save_model(args, exe, iter_num)
                else:
                    save_model(args, exe, iter_num)
        end_time = time.time()
        if 'ce_mode' in os.environ:
            print("kpis	train_duration	%f" % (end_time - start_time))
        # Postprocess benchmark data
        latencies = batch_times[args.skip_batch_num:]
        latency_avg = np.average(latencies)
        latency_pc99 = np.percentile(latencies, 99)
        fpses = np.divide(args.batch_size, latencies)
        fps_avg = np.average(fpses)
        fps_pc99 = np.percentile(fpses, 1)

        # Benchmark output
        print('\nTotal examples (incl. warm-up): %d' %
              (iter_num * args.batch_size))
        print('average latency: %.5f s, 99pc latency: %.5f s' % (latency_avg,
                                                                 latency_pc99))
        print('average fps: %.5f, fps for 99pc latency: %.5f' % (fps_avg,
                                                                 fps_pc99))


def main():
    args = parser.parse_args()
    print_arguments(args)
    check_gpu(args.use_gpu)
    if args.profile:
        if args.use_gpu:
            with profiler.cuda_profiler("cuda_profiler.txt", 'csv') as nvprof:
                train(args)
        else:
            with profiler.profiler("CPU", sorted_key='total') as cpuprof:
                train(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
