"""Trainer for OCR CTC model."""
import paddle.fluid as fluid
from utility import add_arguments, print_arguments, to_lodtensor, get_feeder_data
from crnn_ctc_model import ctc_train_net
import ctc_reader
import argparse
import functools
import sys
import time
import os

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('batch_size',        int,   32,         "Minibatch size.")
add_arg('pass_num',          int,   100,        "Number of training epochs.")
add_arg('log_period',        int,   1000,       "Log period.")
add_arg('save_model_period', int,   15000,      "Save model period. '-1' means never saving the model.")
add_arg('eval_period',       int,   15000,      "Evaluate period. '-1' means never evaluating the model.")
add_arg('save_model_dir',    str,   "./models", "The directory the model to be saved to.")
add_arg('init_model',        str,   None,       "The init model file of directory.")
add_arg('learning_rate',     float, 1.0e-3,    "Learning rate.")
add_arg('l2',                float, 0.0004,    "L2 regularizer.")
add_arg('momentum',          float, 0.9,       "Momentum.")
add_arg('rnn_hidden_size',   int,   200,       "Hidden size of rnn layers.")
add_arg('use_gpu',           bool,  True,      "Whether use GPU to train.")
add_arg('min_average_window',int,   10000,     "Min average window.")
add_arg('max_average_window',int,   15625,     "Max average window. It is proposed to be set as the number of minibatch in a pass.")
add_arg('average_window',    float, 0.15,      "Average window.")
add_arg('parallel',          bool,  False,     "Whether use parallel training.")
add_arg('train_images',      str,   None,    "The directory of training images."
        "None means using the default training images of reader.")
add_arg('train_list',        str,   None,    "The list file of training images."
        "None means using the default train_list file of reader.")
add_arg('test_images',      str,    None,    "The directory of training images."
        "None means using the default test images of reader.")
add_arg('test_list',        str,    None,   "The list file of training images."
        "None means using the default test_list file of reader.")
add_arg('num_classes',      int,    None,      "The number of classes."
        "None means using the default num_classes from reader.")
# yapf: enable


def train(args, data_reader=ctc_reader):
    """OCR CTC training"""
    num_classes = data_reader.num_classes(
    ) if args.num_classes is None else args.num_classes
    data_shape = data_reader.data_shape()
    # define network
    images = fluid.layers.data(name='pixel', shape=data_shape, dtype='float32')
    label = fluid.layers.data(
        name='label', shape=[1], dtype='int32', lod_level=1)
    sum_cost, error_evaluator, inference_program, model_average = ctc_train_net(
        images, label, args, num_classes)

    # data reader
    train_reader = data_reader.train(
        args.batch_size,
        train_images_dir=args.train_images,
        train_list_file=args.train_list)
    test_reader = data_reader.test(
        test_images_dir=args.test_images, test_list_file=args.test_list)

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

    for pass_id in range(args.pass_num):
        error_evaluator.reset(exe)
        batch_id = 1
        total_loss = 0.0
        total_seq_error = 0.0
        # train a pass
        for data in train_reader():
            batch_loss, _, batch_seq_error = exe.run(
                fluid.default_main_program(),
                feed=get_feeder_data(data, place),
                fetch_list=[sum_cost] + error_evaluator.metrics)
            total_loss += batch_loss[0]
            total_seq_error += batch_seq_error[0]
            # training log
            if batch_id % args.log_period == 0:
                print "\nTime: %s; Pass[%d]-batch[%d]; Avg Warp-CTC loss: %s; Avg seq error: %s." % (
                    time.time(), pass_id, batch_id,
                    total_loss / (batch_id * args.batch_size),
                    total_seq_error / (batch_id * args.batch_size))
                sys.stdout.flush()
            # evaluate
            if batch_id % args.eval_period == 0:
                with model_average.apply(exe):
                    error_evaluator.reset(exe)
                    for data in test_reader():
                        exe.run(inference_program,
                                feed=get_feeder_data(data, place))
                    _, test_seq_error = error_evaluator.eval(exe)

                    print "\nTime: %s; Pass[%d]-batch[%d]; Test seq error: %s.\n" % (
                        time.time(), pass_id, batch_id, str(test_seq_error[0]))
            # save model
            if batch_id % args.save_model_period == 0:
                with model_average.apply(exe):
                    filename = "model_%05d_%d" % (pass_id, batch_id)
                    fluid.io.save_params(
                        exe, dirname=args.save_model_dir, filename=filename)
                    print "Saved model to: %s/%s." % (args.save_model_dir,
                                                      filename)

            batch_id += 1


def main():
    args = parser.parse_args()
    print_arguments(args)
    train(args, data_reader=ctc_reader)


if __name__ == "__main__":
    main()
