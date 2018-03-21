"""Trainer for OCR CTC model."""
import paddle.fluid as fluid
import dummy_reader
import ctc_reader
import argparse
from load_model import load_param
import functools
import sys
from utility import add_arguments, print_arguments, to_lodtensor, get_feeder_data
from crnn_ctc_model import ctc_train_net

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('batch_size',     int,   32,     "Minibatch size.")
add_arg('pass_num',       int,   100,     "# of training epochs.")
add_arg('log_period',     int,   1000,   "Log period.")
add_arg('learning_rate',  float, 1.0e-3, "Learning rate.")
add_arg('l2',             float, 0.0004, "L2 regularizer.")
add_arg('max_clip',       float, 10.0,   "Max clip threshold.")
add_arg('min_clip',       float, -10.0,  "Min clip threshold.")
add_arg('momentum',       float, 0.9,    "Momentum.")
add_arg('rnn_hidden_size',int,   200,    "Hidden size of rnn layers.")
add_arg('device',         int,   0,      "Device id.'-1' means running on CPU"
                                         "while '0' means GPU-0.")
add_arg('parallel',     bool,   True,     "Whether use parallel training.")
# yapf: disable

def load_parameter(place):
    params = load_param('./name.map', './data/model/results_without_avg_window/pass-00000/')
    for name in params:
        t = fluid.global_scope().find_var(name).get_tensor()
        t.set(params[name], place)


def train(args, data_reader=dummy_reader):
    """OCR CTC training"""
    num_classes = data_reader.num_classes()
    data_shape = data_reader.data_shape()
    # define network
    images = fluid.layers.data(name='pixel', shape=data_shape, dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int32', lod_level=1)
    sum_cost, error_evaluator, inference_program = ctc_train_net(images, label, args, num_classes)

    # data reader
    train_reader = data_reader.train(args.batch_size)
    test_reader = data_reader.test()
    # prepare environment
    place = fluid.CPUPlace()
    if args.device >= 0:
        place = fluid.CUDAPlace(args.device)
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())
    #load_parameter(place)

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
            if batch_id % 10 == 1:
                print '.',
                sys.stdout.flush()
            if batch_id % args.log_period == 1:
                print "\nPass[%d]-batch[%d]; Avg Warp-CTC loss: %s; Avg seq error: %s." % (
                    pass_id, batch_id, total_loss / (batch_id * args.batch_size), total_seq_error / (batch_id * args.batch_size))
                sys.stdout.flush()
            batch_id += 1

        error_evaluator.reset(exe)
        for data in test_reader():
            exe.run(inference_program, feed=get_feeder_data(data, place))
        _, test_seq_error = error_evaluator.eval(exe)
        print "\nEnd pass[%d]; Test seq error: %s.\n" % (
            pass_id, str(test_seq_error[0]))

def main():
    args = parser.parse_args()
    print_arguments(args)
    train(args, data_reader=ctc_reader)

if __name__ == "__main__":
    main()
