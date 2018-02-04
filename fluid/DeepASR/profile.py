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
        help='Number of lstmp layers to stack. (default: %(default)d)')
    parser.add_argument(
        '--proj_dim',
        type=int,
        default=512,
        help='Project size of lstmp unit. (default: %(default)d)')
    parser.add_argument(
        '--hidden_dim',
        type=int,
        default=1024,
        help='Hidden size of lstmp unit. (default: %(default)d)')
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
    parser.add_argument(
        '--max_batch_num',
        type=int,
        default=11,
        help='Maximum number of batches for profiling. (default: %(default)d)')
    parser.add_argument(
        '--num_batch_to_skip',
        type=int,
        default=1,
        help='Number of batches to skip for profiling. (default: %(default)d)')
    parser.add_argument(
        '--print_train_acc',
        action='store_true',
        help='If set, output training accuray.')
    parser.add_argument(
        '--sorted_key',
        type=str,
        default='total',
        choices=['None', 'total', 'calls', 'min', 'max', 'ave'],
        help='Different types of time to sort the profiling report. '
        '(default: %(default)s)')
    args = parser.parse_args()
    return args


def profile(args):
    """profile the training process"""

    if not args.num_batch_to_skip < args.max_batch_num:
        raise ValueError("arg 'num_batch_to_skip' must be smaller than "
                         "'max_batch_num'.")
    if not args.num_batch_to_skip >= 0:
        raise ValueError("arg 'num_batch_to_skip' must not be smaller than 0.")

    prediction, label, avg_cost = stacked_lstmp_model(
        args.hidden_dim, args.proj_dim, args.stacked_num)

    adam_optimizer = fluid.optimizer.Adam(learning_rate=args.learning_rate)
    adam_optimizer.minimize(avg_cost)

    accuracy = fluid.evaluator.Accuracy(input=prediction, label=label)

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

    sorted_key = None if args.sorted_key is 'None' else args.sorted_key
    with profiler.profiler(args.device, sorted_key) as prof:
        frames_seen, start_time = 0, 0.0
        accuracy.reset(exe)
        for batch_id in range(0, args.max_batch_num):
            if args.num_batch_to_skip == batch_id:
                profiler.reset_profiler()
                start_time = time.time()
                frames_seen = 0
            # load_data
            one_batch = data_reader.get_one_batch(args.batch_size)
            if one_batch == None:
                break
            (bat_feature, bat_label, lod) = one_batch
            res_feature.set(bat_feature, place)
            res_feature.set_lod([lod])
            res_label.set(bat_label, place)
            res_label.set_lod([lod])

            frames_seen += lod[-1]

            outs = exe.run(fluid.default_main_program(),
                           feed={"feature": res_feature,
                                 "label": res_label},
                           fetch_list=[avg_cost] + accuracy.metrics,
                           return_numpy=False)

            if args.print_train_acc:
                print("Batch %d acc: %f" %
                      (batch_id, lodtensor_to_ndarray(outs[1])[0]))
            else:
                sys.stdout.write('.')
                sys.stdout.flush()

        time_consumed = time.time() - start_time
        frames_per_sec = frames_seen / time_consumed
        print("\nTime consumed: %f s, performance: %f frames/s." %
              (time_consumed, frames_per_sec))


if __name__ == '__main__':
    args = parse_args()
    print_arguments(args)
    profile(args)
