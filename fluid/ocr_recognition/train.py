"""Trainer for OCR CTC or attention model."""
import paddle.fluid as fluid
from utility import add_arguments, print_arguments, to_lodtensor, get_ctc_feeder_data, get_attention_feeder_data
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
add_arg('batch_size',        int,   32,         "Minibatch size.")
add_arg('total_step',        int,   720000,    "Number of training iterations.")
add_arg('log_period',        int,   1000,       "Log period.")
add_arg('save_model_period', int,   15000,      "Save model period. '-1' means never saving the model.")
add_arg('eval_period',       int,   15000,      "Evaluate period. '-1' means never evaluating the model.")
add_arg('save_model_dir',    str,   "./models", "The directory the model to be saved to.")
add_arg('model',    str,   "crnn_ctc",           "Which type of network to be used. 'crnn_ctc' or 'attention'")
add_arg('init_model',        str,   None,       "The init model file of directory.")
add_arg('use_gpu',           bool,  True,      "Whether use GPU to train.")
add_arg('min_average_window',int,   10000,     "Min average window.")
add_arg('max_average_window',int,   12500,     "Max average window. It is proposed to be set as the number of minibatch in a pass.")
add_arg('average_window',    float, 0.15,      "Average window.")
add_arg('parallel',          bool,  False,     "Whether use parallel training.")
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
    train_images = None
    train_list = None
    test_images = None
    test_list = None
    num_classes = data_reader.num_classes(
    ) if num_classes is None else num_classes
    data_shape = data_reader.data_shape()
    # define network
    sum_cost, error_evaluator, inference_program, model_average = train_net(
        args, data_shape, num_classes)

    # data reader
    train_reader = data_reader.train(
        args.batch_size,
        train_images_dir=train_images,
        train_list_file=train_list,
        model=args.model)
    test_reader = data_reader.test(
        test_images_dir=test_images, test_list_file=test_list, model=args.model)

    # prepare environment
    place = fluid.CPUPlace()
    if args.use_gpu:
        place = fluid.CUDAPlace(0)
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    # load init model
    if args.init_model is not None:
        model_dir = args.init_model
        model_file_name = None
        if not os.path.isdir(args.init_model):
            model_dir = os.path.dirname(args.init_model)
            model_file_name = os.path.basename(args.init_model)
        fluid.io.load_params(exe, dirname=model_dir, filename=model_file_name)
        print "Init model from: %s." % args.init_model

    train_exe = exe
    error_evaluator.reset(exe)
    if args.parallel:
        train_exe = fluid.ParallelExecutor(
            use_cuda=True, loss_name=sum_cost.name)

    fetch_vars = [sum_cost] + error_evaluator.metrics

    def train_one_batch(data):
        var_names = [var.name for var in fetch_vars]
        if args.parallel:
            results = train_exe.run(var_names,
                                    feed=get_feeder_data(data, place))
            results = [np.array(result).sum() for result in results]
        else:
            results = exe.run(feed=get_feeder_data(data, place),
                              fetch_list=fetch_vars)
            results = [result[0] for result in results]
        return results

    def test(iter_num):
        error_evaluator.reset(exe)
        for data in test_reader():
            exe.run(inference_program, feed=get_feeder_data(data, place))
        _, test_seq_error = error_evaluator.eval(exe)
        print "\nTime: %s; Iter[%d]; Test seq error: %s.\n" % (
            time.time(), iter_num, str(test_seq_error[0]))

    def save_model(args, exe, iter_num):
        filename = "model_%05d" % iter_num
        fluid.io.save_params(
            exe, dirname=args.save_model_dir, filename=filename)
        print "Saved model to: %s/%s." % (args.save_model_dir, filename)

    iter_num = 0
    while True:
        total_loss = 0.0
        total_seq_error = 0.0
        # train a pass
        for data in train_reader():
            iter_num += 1
            if iter_num > args.total_step:
                return
            results = train_one_batch(data)
            total_loss += results[0]
            total_seq_error += results[2]
            # training log
            if iter_num % args.log_period == 0:
                print "\nTime: %s; Iter[%d]; Avg loss: %.3f; Avg seq err: %.3f" % (
                    time.time(), iter_num,
                    total_loss / (args.log_period * args.batch_size),
                    total_seq_error / (args.log_period * args.batch_size))
                sys.stdout.flush()
                total_loss = 0.0
                total_seq_error = 0.0

            # save model
            if iter_num % args.save_model_period == 0:
                if model_average:
                    with model_average.apply(exe):
                        save_model(args, exe, iter_num)
                else:
                    save_model(args, exe, iter_num)

            # evaluate
            if iter_num % args.eval_period == 0:
                if model_average:
                    with model_average.apply(exe):
                        test(iter_num)
                else:
                    test(iter_num)


def main():
    args = parser.parse_args()
    print_arguments(args)
    train(args)


if __name__ == "__main__":
    main()
