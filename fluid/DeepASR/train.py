from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import numpy as np
import argparse
import time

import paddle.fluid as fluid
import data_utils.augmentor.trans_mean_variance_norm as trans_mean_variance_norm
import data_utils.augmentor.trans_add_delta as trans_add_delta
import data_utils.augmentor.trans_splice as trans_splice
import data_utils.async_data_reader as reader
from data_utils.util import lodtensor_to_ndarray
from model_utils.model import stacked_lstmp_model


def parse_args():
    parser = argparse.ArgumentParser("Training for stacked LSTMP model.")
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
        help="The path for feature's global mean and variance. "
        "(default: %(default)s)")
    parser.add_argument(
        '--train_feature_lst',
        type=str,
        default='data/feature.lst',
        help='The feature list path for training. (default: %(default)s)')
    parser.add_argument(
        '--train_label_lst',
        type=str,
        default='data/label.lst',
        help='The label list path for training. (default: %(default)s)')
    parser.add_argument(
        '--val_feature_lst',
        type=str,
        default='data/val_feature.lst',
        help='The feature list path for validation. (default: %(default)s)')
    parser.add_argument(
        '--val_label_lst',
        type=str,
        default='data/val_label.lst',
        help='The label list path for validation. (default: %(default)s)')
    parser.add_argument(
        '--init_model_path',
        type=str,
        default=None,
        help="The model (checkpoint) path which the training resumes from. "
        "If None, train the model from scratch. (default: %(default)s)")
    parser.add_argument(
        '--checkpoints',
        type=str,
        default='./checkpoints',
        help="The directory for saving checkpoints. Do not save checkpoints "
        "if set to ''. (default: %(default)s)")
    parser.add_argument(
        '--infer_models',
        type=str,
        default='./infer_models',
        help="The directory for saving inference models. Do not save inference "
        "models if set to ''. (default: %(default)s)")
    args = parser.parse_args()
    return args


def print_arguments(args):
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(vars(args).iteritems()):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')


def train(args):
    """train in loop.
    """

    # paths check
    if args.init_model_path is not None and \
            not os.path.exists(args.init_model_path):
        raise IOError("Invalid initial model path!")
    if args.checkpoints != '' and not os.path.exists(args.checkpoints):
        os.mkdir(args.checkpoints)
    if args.infer_models != '' and not os.path.exists(args.infer_models):
        os.mkdir(args.infer_models)

    prediction, avg_cost, accuracy = stacked_lstmp_model(
        frame_dim=args.frame_dim,
        hidden_dim=args.hidden_dim,
        proj_dim=args.proj_dim,
        stacked_num=args.stacked_num,
        class_num=args.class_num,
        parallel=args.parallel)

    # program for test
    test_program = fluid.default_main_program().clone()

    optimizer = fluid.optimizer.Adam(learning_rate=args.learning_rate)
    optimizer.minimize(avg_cost)

    place = fluid.CPUPlace() if args.device == 'CPU' else fluid.CUDAPlace(0)
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    # resume training if initial model provided.
    if args.init_model_path is not None:
        fluid.io.load_persistables(exe, args.init_model_path)

    ltrans = [
        trans_add_delta.TransAddDelta(2, 2),
        trans_mean_variance_norm.TransMeanVarianceNorm(args.mean_var),
        trans_splice.TransSplice()
    ]

    feature_t = fluid.LoDTensor()
    label_t = fluid.LoDTensor()

    # validation
    def test(exe):
        # If test data not found, return invalid cost and accuracy
        if not (os.path.exists(args.val_feature_lst) and
                os.path.exists(args.val_label_lst)):
            return -1.0, -1.0
        # test data reader
        test_data_reader = reader.AsyncDataReader(args.val_feature_lst,
                                                  args.val_label_lst)
        test_data_reader.set_transformers(ltrans)
        test_costs, test_accs = [], []
        for batch_id, batch_data in enumerate(
                test_data_reader.batch_iterator(args.batch_size,
                                                args.minimum_batch_size)):
            # load_data
            (features, labels, lod, _) = batch_data
            feature_t.set(features, place)
            feature_t.set_lod([lod])
            label_t.set(labels, place)
            label_t.set_lod([lod])

            cost, acc = exe.run(test_program,
                                feed={"feature": feature_t,
                                      "label": label_t},
                                fetch_list=[avg_cost, accuracy],
                                return_numpy=False)
            test_costs.append(lodtensor_to_ndarray(cost)[0])
            test_accs.append(lodtensor_to_ndarray(acc)[0])
        return np.mean(test_costs), np.mean(test_accs)

    # train data reader
    train_data_reader = reader.AsyncDataReader(args.train_feature_lst,
                                               args.train_label_lst, -1)

    train_data_reader.set_transformers(ltrans)
    # train
    for pass_id in xrange(args.pass_num):
        pass_start_time = time.time()
        for batch_id, batch_data in enumerate(
                train_data_reader.batch_iterator(args.batch_size,
                                                 args.minimum_batch_size)):
            # load_data
            (features, labels, lod, name_lst) = batch_data
            feature_t.set(features, place)
            feature_t.set_lod([lod])
            label_t.set(labels, place)
            label_t.set_lod([lod])

            to_print = batch_id > 0 and (batch_id % args.print_per_batches == 0)
            outs = exe.run(fluid.default_main_program(),
                           feed={"feature": feature_t,
                                 "label": label_t},
                           fetch_list=[avg_cost, accuracy] if to_print else [],
                           return_numpy=False)

            if to_print:
                print("\nBatch %d, train cost: %f, train acc: %f" %
                      (batch_id, lodtensor_to_ndarray(outs[0])[0],
                       lodtensor_to_ndarray(outs[1])[0]))
                # save the latest checkpoint
                if args.checkpoints != '':
                    model_path = os.path.join(args.checkpoints,
                                              "deep_asr.latest.checkpoint")
                    fluid.io.save_persistables(exe, model_path)
            else:
                sys.stdout.write('.')
                sys.stdout.flush()
        # run test
        val_cost, val_acc = test(exe)

        # save checkpoint per pass
        if args.checkpoints != '':
            model_path = os.path.join(
                args.checkpoints,
                "deep_asr.pass_" + str(pass_id) + ".checkpoint")
            fluid.io.save_persistables(exe, model_path)
        # save inference model
        if args.infer_models != '':
            model_path = os.path.join(
                args.infer_models,
                "deep_asr.pass_" + str(pass_id) + ".infer.model")
            fluid.io.save_inference_model(model_path, ["feature"],
                                          [prediction], exe)
        # cal pass time
        pass_end_time = time.time()
        time_consumed = pass_end_time - pass_start_time
        # print info at pass end
        print("\nPass %d, time consumed: %f s, val cost: %f, val acc: %f\n" %
              (pass_id, time_consumed, val_cost, val_acc))


if __name__ == '__main__':
    args = parse_args()
    print_arguments(args)

    train(args)
