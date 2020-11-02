from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import numpy as np
import argparse
import time

import paddle.fluid as fluid
import paddle.fluid.profiler as profiler
import _init_paths
import data_utils.augmentor.trans_mean_variance_norm as trans_mean_variance_norm
import data_utils.augmentor.trans_add_delta as trans_add_delta
import data_utils.augmentor.trans_splice as trans_splice
import data_utils.augmentor.trans_delay as trans_delay
import data_utils.async_data_reader as reader
from model_utils.model import stacked_lstmp_model
from data_utils.util import lodtensor_to_ndarray


def parse_args():
    parser = argparse.ArgumentParser("Profiling for the stacked LSTMP model.")
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='The sequence number of a batch data. (default: %(default)d)')
    parser.add_argument(
        '--minimum_batch_size',
        type=int,
        default=1,
        help='The minimum sequence number of a batch data. '
        '(default: %(default)d)')
    parser.add_argument(
        '--frame_dim',
        type=int,
        default=120 * 11,
        help='Frame dimension of feature data. (default: %(default)d)')
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
        '--class_num',
        type=int,
        default=1749,
        help='Number of classes in label. (default: %(default)d)')
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.00016,
        help='Learning rate used to train. (default: %(default)f)')
    parser.add_argument(
        '--device',
        type=str,
        default='GPU',
        choices=['CPU', 'GPU'],
        help='The device type. (default: %(default)s)')
    parser.add_argument(
        '--parallel', action='store_true', help='If set, run in parallel.')
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
        '--first_batches_to_skip',
        type=int,
        default=1,
        help='Number of first batches to skip for profiling. '
        '(default: %(default)d)')
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


def print_arguments(args):
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(vars(args).iteritems()):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')


def profile(args):
    """profile the training process.
    """

    if not args.first_batches_to_skip < args.max_batch_num:
        raise ValueError("arg 'first_batches_to_skip' must be smaller than "
                         "'max_batch_num'.")
    if not args.first_batches_to_skip >= 0:
        raise ValueError(
            "arg 'first_batches_to_skip' must not be smaller than 0.")

    _, avg_cost, accuracy = stacked_lstmp_model(
        frame_dim=args.frame_dim,
        hidden_dim=args.hidden_dim,
        proj_dim=args.proj_dim,
        stacked_num=args.stacked_num,
        class_num=args.class_num,
        parallel=args.parallel)

    optimizer = fluid.optimizer.Adam(
        learning_rate=fluid.layers.exponential_decay(
            learning_rate=args.learning_rate,
            decay_steps=1879,
            decay_rate=1 / 1.2,
            staircase=True))
    optimizer.minimize(avg_cost)

    place = fluid.CPUPlace() if args.device == 'CPU' else fluid.CUDAPlace(0)
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    ltrans = [
        trans_add_delta.TransAddDelta(2, 2),
        trans_mean_variance_norm.TransMeanVarianceNorm(args.mean_var),
        trans_splice.TransSplice(5, 5), trans_delay.TransDelay(5)
    ]

    data_reader = reader.AsyncDataReader(
        args.feature_lst, args.label_lst, -1, split_sentence_threshold=1024)
    data_reader.set_transformers(ltrans)

    feature_t = fluid.LoDTensor()
    label_t = fluid.LoDTensor()

    sorted_key = None if args.sorted_key is 'None' else args.sorted_key
    with profiler.profiler(args.device, sorted_key) as prof:
        frames_seen, start_time = 0, 0.0
        for batch_id, batch_data in enumerate(
                data_reader.batch_iterator(args.batch_size,
                                           args.minimum_batch_size)):
            if batch_id >= args.max_batch_num:
                break
            if args.first_batches_to_skip == batch_id:
                profiler.reset_profiler()
                start_time = time.time()
                frames_seen = 0
            # load_data
            (features, labels, lod, _) = batch_data
            features = np.reshape(features, (-1, 11, 3, args.frame_dim))
            features = np.transpose(features, (0, 2, 1, 3))
            feature_t.set(features, place)
            feature_t.set_lod([lod])
            label_t.set(labels, place)
            label_t.set_lod([lod])

            frames_seen += lod[-1]

            outs = exe.run(fluid.default_main_program(),
                           feed={"feature": feature_t,
                                 "label": label_t},
                           fetch_list=[avg_cost, accuracy]
                           if args.print_train_acc else [],
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
    import paddle
    paddle.enable_static()
    args = parse_args()
    print_arguments(args)
    profile(args)
