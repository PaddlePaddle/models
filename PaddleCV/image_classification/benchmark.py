from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import numpy as np
import time
import sys
import functools
import math
import paddle
import paddle.fluid as fluid
import reader as reader
import argparse
import functools
import subprocess
import utils
import models
from utils.utility import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('batch_size',       int,   256,                   "Minibatch size.")
add_arg('use_gpu',          bool,  True,                 "Whether to use GPU or not.")
add_arg('total_images',     int,   50000,              "Training image number.")
add_arg('num_epochs',       int,   2,                  "number of epochs.")
add_arg('class_dim',        int,   1000,                 "Class number.")
add_arg('image_shape',      str,   "3,224,224",          "input image size")
add_arg('with_mem_opt',     bool,  True,                 "Whether to use memory optimization or not.")
add_arg('lr',               float, 0.1,                  "set learning rate.")
add_arg('model',            str,   "ResNet50",           "Set the network to use.")
add_arg('data_dir',         str,   "./data/ILSVRC2012",  "The ImageNet dataset root dir.")
add_arg('skip_steps',       int,   10,                   "Skip initial steps to count")
def optimizer_setting(params):
    lr = params["lr"]
    optimizer = fluid.optimizer.Momentum(
        learning_rate=lr,
        momentum=0.9,
        regularization=fluid.regularizer.L2Decay(1e-4))
    return optimizer

def net_config(image, label, model, args):
    model_list = [m for m in dir(models) if "__" not in m]
    assert args.model in model_list, "{} is not lists: {}".format(args.model,
                                                                  model_list)
    class_dim = args.class_dim
    model_name = args.model

    if model_name == "GoogleNet":
        out0, out1, out2 = model.net(input=image, class_dim=class_dim)
        cost0 = fluid.layers.cross_entropy(input=out0, label=label)
        cost1 = fluid.layers.cross_entropy(input=out1, label=label)
        cost2 = fluid.layers.cross_entropy(input=out2, label=label)
        avg_cost0 = fluid.layers.mean(x=cost0)
        avg_cost1 = fluid.layers.mean(x=cost1)
        avg_cost2 = fluid.layers.mean(x=cost2)

        avg_cost = avg_cost0 + 0.3 * avg_cost1 + 0.3 * avg_cost2
        acc_top1 = fluid.layers.accuracy(input=out0, label=label, k=1)
        acc_top5 = fluid.layers.accuracy(input=out0, label=label, k=5)
    else:
        out = model.net(input=image, class_dim=class_dim)
        cost, pred = fluid.layers.softmax_with_cross_entropy(
            out, label, return_softmax=True)
        avg_cost = fluid.layers.mean(x=cost)
        acc_top1 = fluid.layers.accuracy(input=pred, label=label, k=1)
        acc_top5 = fluid.layers.accuracy(input=pred, label=label, k=5)

    return avg_cost, acc_top1, acc_top5

def build_program (main_prog, startup_prog, args):
    image_shape = [int(m) for m in args.image_shape.split(",")]
    model_name = args.model
    model_list = [m for m in dir(models) if "__" not in m]
    assert model_name in model_list, "{} is not in lists: {}".format(args.model,
                                                                     model_list)
    model = models.__dict__[model_name]()
    with fluid.program_guard(main_prog, startup_prog):
        py_reader = fluid.layers.py_reader(
            capacity=16,
            shapes=[[-1] + image_shape, [-1, 1]],
            lod_levels=[0, 0],
            dtypes=["float32", "int64"],
            use_double_buffer=True)
        with fluid.unique_name.guard():
            image, label = fluid.layers.read_file(py_reader)
            avg_cost, acc_top1, acc_top5 = net_config(image, label, model, args)
            params = model.params
            params["total_images"] = args.total_images
            params["lr"] = args.lr
            params["num_epochs"] = args.num_epochs

            optimizer = optimizer_setting(params)
            optimizer.minimize(avg_cost)
            global_lr = optimizer._global_learning_rate()

    return py_reader, avg_cost, acc_top1, acc_top5, global_lr

def get_device_num():
    visible_device = os.getenv('CUDA_VISIBLE_DEVICES')
    if visible_device:
        device_num = len(visible_device.split(','))
    else:
        device_num = subprocess.check_output(['nvidia-smi','-L']).decode().count('\n')
    return device_num

def train(args):
    # parameters from arguments
    model_name = args.model
    skip_steps = args.skip_steps
    with_memory_optimization = args.with_mem_opt

    startup_prog = fluid.Program()
    train_prog = fluid.Program()

    train_py_reader, train_cost, train_acc1, train_acc5, global_lr = build_program(
        main_prog=train_prog,
        startup_prog=startup_prog,
        args=args)

    if with_memory_optimization:
        fluid.memory_optimize(train_prog)
    
    total_images = args.total_images
    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(startup_prog)

    if args.use_gpu:
        device_num = get_device_num()
    else:
        device_num = 1
    batch_size = args.batch_size
    train_batch_size = args.batch_size / device_num

    train_reader = paddle.batch(
            reader.train(), batch_size=train_batch_size, drop_last=True)

    train_py_reader.decorate_paddle_reader(train_reader)

    train_exe = fluid.ParallelExecutor(
        main_program=train_prog,
        use_cuda=bool(args.use_gpu),
        loss_name=train_cost.name)

    train_fetch_list = [
        train_cost.name, train_acc1.name, train_acc5.name, global_lr.name
    ]
    no_feed_data = os.getenv('FLAGS_reader_queue_speed_test_mode=True')
    params = models.__dict__[args.model]().params
    for pass_id in range(params["num_epochs"]):

        train_py_reader.start()

        train_info = [[], [], []]
        train_time = []
        total_time=0
        batch_id = 0
        try:
            while True:
                t1 = time.time()

                loss, acc1, acc5, lr = train_exe.run(
                    fetch_list=train_fetch_list)
                t2 = time.time()
                period = t2 - t1
                if batch_id > skip_steps-1:
                  total_time +=period
                
                loss = np.mean(np.array(loss))

                acc1 = np.mean(np.array(acc1))
                acc5 = np.mean(np.array(acc5))
                train_info[0].append(loss)
                train_info[1].append(acc1)
                train_info[2].append(acc5)
                lr = np.mean(np.array(lr))
                train_time.append(period)

                if batch_id % 10 == 0:
                    print("Pass {0}, trainbatch {1}[{2}], time {3}"
                          .format(pass_id, batch_id, int(math.floor(total_images/batch_size)),"%2.5f sec" % period))
                    sys.stdout.flush()
                batch_id += 1
                if no_feed_data and batch_id== 50:
                    print("===================================")
                    print("No feed data")
                    print("GPU:", device_num)
                    print("batch_size:", batch_size)
                    print("speed:", batch_size/period, "images/s")
                    print("===================================")
                    exit(0)

                        
        except fluid.core.EOFException:
            train_py_reader.reset()
        print("=================================")
        print("Pass", pass_id)
        print("GPU:", device_num)
        print("Skip first", skip_steps," steps:")
        print("Elapsed time:", total_time)
        print("The number of batch:", batch_id-skip_steps)
        print("batch size: ", batch_size)

        print("speed:", (batch_size*(batch_id-skip_steps))/total_images," images/s")
        print("=================================")

        sys.stdout.flush()

def main():
    args = parser.parse_args()
    print_arguments(args)
    train(args)

if __name__ == '__main__':
    main()
