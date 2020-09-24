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
import data_utils.augmentor.trans_delay as trans_delay
import data_utils.async_data_reader as reader
from model_utils.model import stacked_lstmp_model


def parse_args():
    parser = argparse.ArgumentParser("Training for stacked LSTMP model.")
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='The sequence number of a batch data. Batch size per GPU. (default: %(default)d)'
    )
    parser.add_argument(
        '--minimum_batch_size',
        type=int,
        default=1,
        help='The minimum sequence number of a batch data. '
        '(default: %(default)d)')
    parser.add_argument(
        '--frame_dim',
        type=int,
        default=80,
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
        default=3040,
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

    train_program = fluid.Program()
    train_startup = fluid.Program()

    input_fields = {
        'names': ['feature', 'label'],
        'shapes': [[None, 3, 11, args.frame_dim], [None, 1]],
        'dtypes': ['float32', 'int64'],
        'lod_levels': [1, 1]
    }

    with fluid.program_guard(train_program, train_startup):
        with fluid.unique_name.guard():

            inputs = [
                fluid.data(
                    name=input_fields['names'][i],
                    shape=input_fields['shapes'][i],
                    dtype=input_fields['dtypes'][i],
                    lod_level=input_fields['lod_levels'][i])
                for i in range(len(input_fields['names']))
            ]

            train_reader = fluid.io.DataLoader.from_generator(
                feed_list=inputs,
                capacity=64,
                iterable=False,
                use_double_buffer=True)

            (feature, label) = inputs

            prediction, avg_cost, accuracy = stacked_lstmp_model(
                feature=feature,
                label=label,
                hidden_dim=args.hidden_dim,
                proj_dim=args.proj_dim,
                stacked_num=args.stacked_num,
                class_num=args.class_num)
            # optimizer = fluid.optimizer.Momentum(learning_rate=args.learning_rate, momentum=0.9)
            optimizer = fluid.optimizer.Adam(
                learning_rate=fluid.layers.exponential_decay(
                    learning_rate=args.learning_rate,
                    decay_steps=1879,
                    decay_rate=1 / 1.2,
                    staircase=True))
            optimizer.minimize(avg_cost)

    test_program = fluid.Program()
    test_startup = fluid.Program()
    with fluid.program_guard(test_program, test_startup):
        with fluid.unique_name.guard():
            inputs = [
                fluid.data(
                    name=input_fields['names'][i],
                    shape=input_fields['shapes'][i],
                    dtype=input_fields['dtypes'][i],
                    lod_level=input_fields['lod_levels'][i])
                for i in range(len(input_fields['names']))
            ]

            test_reader = fluid.io.DataLoader.from_generator(
                feed_list=inputs,
                capacity=64,
                iterable=False,
                use_double_buffer=True)

            (feature, label) = inputs
            prediction, avg_cost, accuracy = stacked_lstmp_model(
                feature=feature,
                label=label,
                hidden_dim=args.hidden_dim,
                proj_dim=args.proj_dim,
                stacked_num=args.stacked_num,
                class_num=args.class_num)
    test_program = test_program.clone(for_test=True)
    place = fluid.CPUPlace() if args.device == 'CPU' else fluid.CUDAPlace(0)
    exe = fluid.Executor(place)
    exe.run(train_startup)
    exe.run(test_startup)

    if args.parallel:
        exec_strategy = fluid.ExecutionStrategy()
        exec_strategy.num_iteration_per_drop_scope = 10
        train_exe = fluid.ParallelExecutor(
            use_cuda=(args.device == 'GPU'),
            loss_name=avg_cost.name,
            exec_strategy=exec_strategy,
            main_program=train_program)
        test_exe = fluid.ParallelExecutor(
            use_cuda=(args.device == 'GPU'),
            main_program=test_program,
            exec_strategy=exec_strategy,
            share_vars_from=train_exe)

    # resume training if initial model provided.
    if args.init_model_path is not None:
        fluid.io.load_persistables(exe, args.init_model_path)

    ltrans = [
        trans_add_delta.TransAddDelta(2, 2),
        trans_mean_variance_norm.TransMeanVarianceNorm(args.mean_var),
        trans_splice.TransSplice(5, 5), trans_delay.TransDelay(5)
    ]

    # bind train_reader
    train_data_reader = reader.AsyncDataReader(
        args.train_feature_lst,
        args.train_label_lst,
        -1,
        split_sentence_threshold=1024)

    train_data_reader.set_transformers(ltrans)

    def train_data_provider():
        for data in train_data_reader.batch_iterator(args.batch_size,
                                                     args.minimum_batch_size):
            yield batch_data_to_lod_tensors(args, data, fluid.CPUPlace())

    train_reader.set_batch_generator(train_data_provider)

    if (os.path.exists(args.val_feature_lst) and
            os.path.exists(args.val_label_lst)):
        # test data reader
        test_data_reader = reader.AsyncDataReader(
            args.val_feature_lst,
            args.val_label_lst,
            -1,
            split_sentence_threshold=1024)
        test_data_reader.set_transformers(ltrans)

        def test_data_provider():
            for data in test_data_reader.batch_iterator(
                    args.batch_size, args.minimum_batch_size):
                yield batch_data_to_lod_tensors(args, data, fluid.CPUPlace())

        test_reader.set_batch_generator(test_data_provider)

    # validation
    def test(exe):
        # If test data not found, return invalid cost and accuracy
        if not (os.path.exists(args.val_feature_lst) and
                os.path.exists(args.val_label_lst)):
            return -1.0, -1.0
        batch_id = 0
        test_costs = []
        test_accs = []
        while True:
            if batch_id == 0:
                test_reader.start()
            try:
                if args.parallel:
                    cost, acc = exe.run(
                        fetch_list=[avg_cost.name, accuracy.name],
                        return_numpy=False)
                else:
                    cost, acc = exe.run(program=test_program,
                                        fetch_list=[avg_cost, accuracy],
                                        return_numpy=False)
                sys.stdout.write('.')
                sys.stdout.flush()
                test_costs.append(np.array(cost)[0])
                test_accs.append(np.array(acc)[0])
                batch_id += 1
            except fluid.core.EOFException:
                test_reader.reset()
                break
        return np.mean(test_costs), np.mean(test_accs)

    # train
    for pass_id in xrange(args.pass_num):
        pass_start_time = time.time()
        batch_id = 0
        while True:
            if batch_id == 0:
                train_reader.start()
            to_print = batch_id > 0 and (batch_id % args.print_per_batches == 0)
            try:
                if args.parallel:
                    outs = train_exe.run(
                        fetch_list=[avg_cost.name, accuracy.name]
                        if to_print else [],
                        return_numpy=False)
                else:
                    outs = exe.run(program=train_program,
                                   fetch_list=[avg_cost, accuracy]
                                   if to_print else [],
                                   return_numpy=False)
            except fluid.core.EOFException:
                train_reader.reset()
                break

            if to_print:
                if args.parallel:
                    print("\nBatch %d, train cost: %f, train acc: %f" %
                          (batch_id, np.mean(outs[0]), np.mean(outs[1])))
                else:
                    print("\nBatch %d, train cost: %f, train acc: %f" % (
                        batch_id, np.array(outs[0])[0], np.array(outs[1])[0]))
                # save the latest checkpoint
                if args.checkpoints != '':
                    model_path = os.path.join(args.checkpoints,
                                              "deep_asr.latest.checkpoint")
                    fluid.io.save_persistables(exe, model_path, train_program)
            else:
                sys.stdout.write('.')
                sys.stdout.flush()

            batch_id += 1
        # run test
        val_cost, val_acc = test(test_exe if args.parallel else exe)

        # save checkpoint per pass
        if args.checkpoints != '':
            model_path = os.path.join(
                args.checkpoints,
                "deep_asr.pass_" + str(pass_id) + ".checkpoint")
            fluid.io.save_persistables(exe, model_path, train_program)
        # save inference model
        if args.infer_models != '':
            model_path = os.path.join(
                args.infer_models,
                "deep_asr.pass_" + str(pass_id) + ".infer.model")
            fluid.io.save_inference_model(model_path, ["feature"],
                                          [prediction], exe, train_program)
        # cal pass time
        pass_end_time = time.time()
        time_consumed = pass_end_time - pass_start_time
        # print info at pass end
        print("\nPass %d, time consumed: %f s, val cost: %f, val acc: %f\n" %
              (pass_id, time_consumed, val_cost, val_acc))


def batch_data_to_lod_tensors(args, batch_data, place):
    features, labels, lod, name_lst = batch_data
    features = np.reshape(features, (-1, 11, 3, args.frame_dim))
    features = np.transpose(features, (0, 2, 1, 3))
    feature_t = fluid.LoDTensor()
    label_t = fluid.LoDTensor()
    feature_t.set(features, place)
    feature_t.set_lod([lod])
    label_t.set(labels, place)
    label_t.set_lod([lod])
    return feature_t, label_t


if __name__ == '__main__':
    import paddle
    paddle.enable_static()
    args = parse_args()
    print_arguments(args)

    train(args)
