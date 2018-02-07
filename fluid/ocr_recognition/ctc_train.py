"""Trainer for OCR CTC model."""
import paddle.v2 as paddle
import paddle.v2.fluid as fluid
import dummy_reader
import argparse
import functools
import sys
from utility import add_arguments, print_arguments, to_lodtensor, get_feeder_data
from crnn_ctc_model import ctc_train_net

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('batch_size',     int,   32,     "Minibatch size.")
add_arg('pass_num',       int,   32,     "# of training epochs.")
add_arg('log_period',     int,   1000,   "Log period.")
add_arg('learning_rate',  float, 1.0e-3, "Learning rate.")
add_arg('l2',             float, 0.0004, "L2 regularizer.")
add_arg('max_clip',       float, 10.0,   "Max clip threshold.")
add_arg('min_clip',       float, -10.0,  "Min clip threshold.")
add_arg('momentum',       float, 0.9,    "Momentum.")
add_arg('rnn_hidden_size',int,   200,    "Hidden size of rnn layers.")
add_arg('device',         int,   0,      "Device id.'-1' means running on CPU"
                                         "while '0' means GPU-0.")
# yapf: disable

def train(args, data_reader=dummy_reader):
    """OCR CTC training"""
    num_classes = data_reader.num_classes()
    data_shape = data_reader.data_shape()
    # define network
    images = fluid.layers.data(name='pixel', shape=data_shape, dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int32', lod_level=1)
    avg_cost, error_evaluator = ctc_train_net(images, label, args, num_classes)
    # data reader
    train_reader = data_reader.train(args.batch_size)
    test_reader = data_reader.test()
    # prepare environment
    place = fluid.CPUPlace()
    if args.device >= 0:
        place = fluid.CUDAPlace(args.device)
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    inference_program = fluid.io.get_inference_program(error_evaluator)
    for pass_id in range(args.pass_num):
        error_evaluator.reset(exe)
        batch_id = 0
        # train a pass
        for data in train_reader():
            loss, batch_edit_distance = exe.run(
                fluid.default_main_program(),
                feed=get_feeder_data(data, place),
                fetch_list=[avg_cost] + error_evaluator.metrics)
            if batch_id % args.log_period == 0:
                print "Pass[%d]-batch[%d]; Loss: %s; Word error: %s." % (
                    pass_id, batch_id, loss[0], batch_edit_distance[0] / float(args.batch_size))
                sys.stdout.flush()
            batch_id += 1

        train_edit_distance = error_evaluator.eval(exe)
        print "End pass[%d]; Train word error: %s.\n" % (
            pass_id, str(train_edit_distance[0]))

        # evaluate model on test data
        error_evaluator.reset(exe)
        for data in test_reader():
            exe.run(inference_program, feed=get_feeder_data(data, place))
        test_edit_distance = error_evaluator.eval(exe)
        print "End pass[%d]; Test word error: %s.\n" % (
            pass_id, str(test_edit_distance[0]))

def main():
    args = parser.parse_args()
    print_arguments(args)
    train(args, data_reader=dummy_reader)

if __name__ == "__main__":
    main()
