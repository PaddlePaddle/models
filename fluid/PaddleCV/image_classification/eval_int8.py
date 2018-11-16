from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import numpy as np
import time
import sys
import paddle
import paddle.fluid as fluid
import models
import reader
import argparse
import functools
from models.learning_rate import cosine_decay
from utility import add_arguments, print_arguments
import paddle.fluid.profiler as profiler

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('batch_size',       int,  256,                 "Minibatch size.")
add_arg('use_gpu',          bool, True,                "Whether to use GPU or not.")
add_arg('class_dim',        int,  1000,                "Class number.")
add_arg('image_shape',      str,  "3,224,224",         "Input image size")
add_arg('with_mem_opt',     bool, True,                "Whether to use memory optimization or not.")
add_arg('pretrained_model', str,  None,                "Whether to use pretrained model.")
add_arg('skip_batch_num',   int,  5,                    "Skip batch num.")
add_arg('use_transpiler',   bool, True,                 'Whether to use transpiler.')
add_arg('use_fake_data',    bool, False,                'If set, use fake data instead of real data.')
add_arg('iterations',	    int,  100,               	'Fake data iterations')
add_arg('profiler',	    bool, False,                'If true, do profiling.')
# yapf: enable

model_list = [m for m in dir(models) if "__" not in m]


def user_data_reader(data):
    '''
    Creates a data reader for user data.
    '''

    def data_reader():
        while True:
            for b in data:
                yield b

    return data_reader


def eval(args):
    # parameters from arguments
    class_dim = args.class_dim
    pretrained_model = args.pretrained_model
    with_memory_optimization = args.with_mem_opt
    skip_batch_num = args.skip_batch_num
    if skip_batch_num >= args.iterations:
       print("Please ensure the skip_batch_num less than iterations.")
       sys.exit(0)
    image_shape = [int(m) for m in args.image_shape.split(",")]


    if with_memory_optimization:
        fluid.memory_optimize(fluid.default_main_program())

    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)

    [infer_program, feed_dict,
     fetch_targets] = fluid.io.load_inference_model(pretrained_model, exe)
    
    program = infer_program.clone()
    if args.use_transpiler:
        inference_transpiler_program = program.clone()
        t = fluid.transpiler.InferenceTranspiler()
        t.transpile(inference_transpiler_program, place)
        program = inference_transpiler_program

    fake_data = [(
        np.random.rand(image_shape[0] * image_shape[1] * image_shape[2]).astype(np.float32),
        np.random.randint(1, class_dim)) for _ in range(1)]

    if args.use_fake_data:
        val_reader = paddle.batch(
            user_data_reader(fake_data), batch_size=args.batch_size)
    else:
        val_reader = paddle.batch(reader.val(), batch_size=args.batch_size)

    test_info = [[], [], []]
    cnt = 0
    periods = []
    for batch_id, data in enumerate(val_reader()):
        if args.use_fake_data:
            data = val_reader().next()
        image = np.array(map(lambda x: x[0].reshape(image_shape), data)).astype(
            "float32")
        label = np.array(map(lambda x: x[1], data)).astype("int64")
        label = label.reshape([-1, 1])

        t1 = time.time()
        loss, acc1, acc5 = exe.run(program, feed={feed_dict[0]: image, feed_dict[1]: label}, fetch_list=fetch_targets)

        t2 = time.time()
        period = t2 - t1
        loss = np.mean(loss)
        acc1 = np.mean(acc1)
        acc5 = np.mean(acc5)
        test_info[0].append(loss * len(data))
        test_info[1].append(acc1 * len(data))
        test_info[2].append(acc5 * len(data))
        periods.append(period)
        cnt += len(data)
        if batch_id % 10 == 0:
            print("Testbatch {0},loss {1}, "
                  "acc1 {2},acc5 {3},time {4}".format(batch_id, loss, acc1, acc5, "%2.2f sec" % period))
            sys.stdout.flush()
        if batch_id == args.iterations - 1:
            break

    test_loss = np.sum(test_info[0]) / cnt
    test_acc1 = np.sum(test_info[1]) / cnt
    test_acc5 = np.sum(test_info[2]) / cnt
    throughput = cnt / np.sum(periods)
    throughput_skip = (cnt-skip_batch_num*args.batch_size) / np.sum(periods[skip_batch_num:])
    latency = np.average(periods)
    latency_skip = np.average(periods[skip_batch_num:])
    print("Test_loss {0}, test_acc1 {1}, test_acc5 {2}".format(
        test_loss, test_acc1, test_acc5))
    sys.stdout.flush()
    print("throughput {0}, throughput_skip {1}, latency {2}, latency_skip {3}".format(
        throughput, throughput_skip, latency, latency_skip))
    sys.stdout.flush()


def main():
    args = parser.parse_args()
    print_arguments(args)
    if args.profiler:
        if args.use_gpu:
            with profiler.cuda_profiler("cuda_profiler.txt", 'csv') as nvprof:
                eval(args)
        else:
            with profiler.profiler("CPU", sorted_key='total') as cpuprof:
                eval(args)
    else:
        eval(args)


if __name__ == '__main__':
    main()
