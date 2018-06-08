import paddle.fluid as fluid
from utility import add_arguments, print_arguments, to_lodtensor, get_feeder_data
from crnn_ctc_model import ctc_train_net
import ctc_reader
import argparse
import functools
import sys
import time
import os
import numpy as np

import paddle.fluid.profiler as profiler

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
add_arg('use_gpu',           bool,  True,       "Whether to use GPU to train.")
add_arg('min_average_window',int,   10000,      "Min average window.")
add_arg('max_average_window',int,   15625,      "Max average window. It is proposed to be set as the number of minibatch in a pass.")
add_arg('average_window',    float, 0.15,       "Average window.")
add_arg('parallel',          bool,  False,      "Whether to use parallel training.")
add_arg('use_mkldnn',        bool,  False,      "Whether to use mkldnn to train.")
add_arg('profile',           bool,  False,      "Whether to use profiling.")
add_arg('skip_batch_num',    int,   0,          "The number of first minibatches to skip as warm-up for better performance test.")
add_arg('iterations',        int,   0,          "The number of iterations. Zero or less means whole training set. More than 0 means the training set might be looped until # of iterations is reached.")
add_arg('skip_test',         bool,  False,      "Whether to skip test phase.")
# yapf: enable


def train(args, data_reader=ctc_reader):
    """OCR CTC training"""
    num_classes = None
    train_images = None
    train_list = None
    test_images = None
    test_list = None
    num_classes = data_reader.num_classes(
    ) if num_classes is None else num_classes
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
        train_images_dir=train_images,
        train_list_file=train_list,
        cycle=args.iterations > 0)
    test_reader = data_reader.test(
        test_images_dir=test_images, test_list_file=test_list)

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
            use_cuda=True if args.use_gpu else False, loss_name=sum_cost.name)

    fetch_vars = [sum_cost] + error_evaluator.metrics

    def train_one_batch(data):
        var_names = [var.name for var in fetch_vars]
        if args.parallel:
            results = train_exe.run(var_names,
                                    feed_dict=get_feeder_data(data, place))
            results = [np.array(result).sum() for result in results]
        else:
            results = train_exe.run(feed=get_feeder_data(data, place),
                                    fetch_list=fetch_vars)
            results = [result[0] for result in results]
        return results

    def test(pass_id, batch_id):
        error_evaluator.reset(exe)
        for data in test_reader():
            exe.run(inference_program, feed=get_feeder_data(data, place))
        _, test_seq_error = error_evaluator.eval(exe)
        print "\nTime: %s; Pass[%d]-batch[%d]; Test seq error: %s.\n" % (
            time.time(), pass_id, batch_id, str(test_seq_error[0]))

    def save_model(args, exe, pass_id, batch_id):
        filename = "model_%05d_%d" % (pass_id, batch_id)
        fluid.io.save_params(
            exe, dirname=args.save_model_dir, filename=filename)
        print "Saved model to: %s/%s." % (args.save_model_dir, filename)

    for pass_id in range(args.pass_num):
        batch_id = 1
        total_loss = 0.0
        total_seq_error = 0.0
        batch_times = []
        iters = 0
        # train a pass
        for data in train_reader():
            if args.iterations > 0 and iters == args.iterations + args.skip_batch_num:
                break
            if iters < args.skip_batch_num:
                print("Warm-up iteration")
            if iters == args.skip_batch_num:
                profiler.reset_profiler()
            start = time.time()
            results = train_one_batch(data)
            batch_time = time.time() - start
            fps = args.batch_size / batch_time
            batch_times.append(batch_time)
            total_loss += results[0]
            total_seq_error += results[2]
            # training log
            if batch_id % args.log_period == 0:
                print "\nTime: %s; Pass[%d]-batch[%d]; Avg Warp-CTC loss: %s; Avg seq err: %s" % (
                    time.time(), pass_id, batch_id,
                    total_loss / (batch_id * args.batch_size),
                    total_seq_error / (batch_id * args.batch_size))
                sys.stdout.flush()

            # evaluate
            if not args.skip_test and batch_id % args.eval_period == 0:
                if model_average:
                    with model_average.apply(exe):
                        test(pass_id, batch_id)
                else:
                    test(pass_id, batch_d)

            # save model
            if batch_id % args.save_model_period == 0:
                if model_average:
                    with model_average.apply(exe):
                        save_model(args, exe, pass_id, batch_id)
                else:
                    save_model(args, exe, pass_id, batch_id)

            batch_id += 1
            iters += 1

        # Postprocess benchmark data
        latencies = batch_times[args.skip_batch_num:]
        latency_avg = np.average(latencies)
        latency_pc99 = np.percentile(latencies, 99)
        fpses = np.divide(args.batch_size, latencies)
        fps_avg = np.average(fpses)
        fps_pc99 = np.percentile(fpses, 1)

        # Benchmark output
        print('\nTotal examples (incl. warm-up): %d' %
              (iters * args.batch_size))
        print('average latency: %.5f s, 99pc latency: %.5f s' % (latency_avg,
                                                                 latency_pc99))
        print('average fps: %.5f, fps for 99pc latency: %.5f' % (fps_avg,
                                                                 fps_pc99))


def main():
    args = parser.parse_args()
    print_arguments(args)
    if args.profile:
        if args.use_gpu:
            with profiler.cuda_profiler("cuda_profiler.txt", 'csv') as nvprof:
                train(args, data_reader=ctc_reader)
        else:
            with profiler.profiler("CPU", sorted_key='total') as cpuprof:
                train(args, data_reader=ctc_reader)
    else:
        train(args, data_reader=ctc_reader)


if __name__ == "__main__":
    main()
