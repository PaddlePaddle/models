from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import numpy as np
import argparse
import time

import paddle.v2 as paddle
import paddle.v2.fluid as fluid
import paddle.v2.fluid.profiler as profiler
import data_utils.trans_mean_variance_norm as trans_mean_variance_norm
import data_utils.trans_add_delta as trans_add_delta
import data_utils.trans_splice as trans_splice
import data_utils.data_reader as reader
from model import stacked_lstmp_model
from utils import print_arguments, lodtensor_to_ndarray


def parse_args():
    parser = argparse.ArgumentParser("LSTM model benchmark.")
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='The sequence number of a batch data. (default: %(default)d)')
    parser.add_argument(
        '--stacked_num',
        type=int,
        default=5,
        help='Number of lstm layers to stack. (default: %(default)d)')
    parser.add_argument(
        '--proj_dim',
        type=int,
        default=512,
        help='Project size of lstm unit. (default: %(default)d)')
    parser.add_argument(
        '--hidden_dim',
        type=int,
        default=1024,
        help='Hidden size of lstm unit. (default: %(default)d)')
    parser.add_argument(
        '--pass_num',
        type=int,
        default=100,
        help='Epoch number to train. (default: %(default)d)')
    parser.add_argument(
        '--print_per_batches',
        type=int,
        default=100,
        help='Interval to print training accuracy. (default: %(default)d)')
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.002,
        help='Learning rate used to train. (default: %(default)f)')
    parser.add_argument(
        '--device',
        type=str,
        default='GPU',
        choices=['CPU', 'GPU'],
        help='The device type. (default: %(default)s)')
    parser.add_argument(
        '--mean_var',
        type=str,
        default='data/global_mean_var_search26kHr',
        help='mean var path')
    parser.add_argument(
        '--feature_lst',
        type=str,
        default='data/feature.lst',
        help='feature list path.')
    parser.add_argument(
        '--label_lst',
        type=str,
        default='data/label.lst',
        help='label list path.')
    args = parser.parse_args()
    return args


def train(args):
    """train in loop."""

    prediction, label, avg_cost = stacked_lstmp_model(
        args.hidden_dim, args.proj_dim, args.stacked_num)

    adam_optimizer = fluid.optimizer.Adam(learning_rate=args.learning_rate)
    adam_optimizer.minimize(avg_cost)

    accuracy = fluid.evaluator.Accuracy(input=prediction, label=label)

    # clone from default main program
    inference_program = fluid.default_main_program().clone()
    with fluid.program_guard(inference_program):
        test_accuracy = fluid.evaluator.Accuracy(input=prediction, label=label)
        test_target = [avg_cost] + test_accuracy.metrics + test_accuracy.states
        inference_program = fluid.io.get_inference_program(test_target)

    place = fluid.CPUPlace() if args.device == 'CPU' else fluid.CUDAPlace(0)
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    ltrans = [
        trans_add_delta.TransAddDelta(2, 2),
        trans_mean_variance_norm.TransMeanVarianceNorm(args.mean_var),
        trans_splice.TransSplice()
    ]

    data_reader = reader.DataRead(args.feature_lst, args.label_lst)
    data_reader.set_trans(ltrans)

    res_feature = fluid.LoDTensor()
    res_label = fluid.LoDTensor()
    for pass_id in xrange(args.pass_num):
        pass_start_time = time.time()
        accuracy.reset(exe)
        batch_id = 0
        while True:
            # load_data
            one_batch = data_reader.get_one_batch(args.batch_size)
            if one_batch == None:
                break
            (bat_feature, bat_label, lod) = one_batch
            res_feature.set(bat_feature, place)
            res_feature.set_lod([lod])
            res_label.set(bat_label, place)
            res_label.set_lod([lod])

            batch_id += 1
            _, acc = exe.run(fluid.default_main_program(),
                             feed={"feature": res_feature,
                                   "label": res_label},
                             fetch_list=[avg_cost] + accuracy.metrics,
                             return_numpy=False)

            if batch_id > 0 and (batch_id % args.print_per_batches == 0):
                print("\nBatch %d, training acc: %f" %
                      (batch_id, lodtensor_to_ndarray(acc)[0]))
            else:
                sys.stdout.write('.')
                sys.stdout.flush()

        pass_end_time = time.time()
        time_consumed = pass_end_time - pass_start_time
        # need to add test logic (kuke)
        print("\nPass %d, time: %fs, test accuracy: 0.0f\n" %
              (pass_id, time_consumed))


if __name__ == '__main__':
    args = parse_args()
    print_arguments(args)

    train(args)
