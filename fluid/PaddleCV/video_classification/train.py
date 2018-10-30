import os
import numpy as np
import time
import sys
import paddle
import paddle.fluid as fluid
from resnet import TSN_ResNet
import reader

import argparse
import functools
from paddle.fluid.framework import Parameter
from utility import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('batch_size',       int,    128,            "Minibatch size.")
add_arg('num_layers',       int,    50,             "How many layers for ResNet model.")
add_arg('with_mem_opt',     bool,   True,           "Whether to use memory optimization or not.")
add_arg('num_epochs',       int,    60,             "Number of epochs.")
add_arg('class_dim',        int,    101,            "Number of class.")
add_arg('seg_num',          int,    7,              "Number of segments.")
add_arg('image_shape',      str,    "3,224,224",    "Input image size.")
add_arg('model_save_dir',   str,    "output",       "Model save directory.")
add_arg('pretrained_model', str,    None,           "Whether to use pretrained model.")
add_arg('total_videos',     int,    9537,           "Training video number.")
add_arg('lr_init',          float,  0.01,           "Set initial learning rate.")
# yapf: enable


def train(args):
    # parameters from arguments
    seg_num = args.seg_num
    class_dim = args.class_dim
    num_layers = args.num_layers
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    pretrained_model = args.pretrained_model
    model_save_dir = args.model_save_dir

    image_shape = [int(m) for m in args.image_shape.split(",")]
    image_shape = [seg_num] + image_shape

    # model definition
    model = TSN_ResNet(layers=num_layers, seg_num=seg_num)

    image = fluid.layers.data(name='image', shape=image_shape, dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')

    out = model.net(input=image, class_dim=class_dim)
    cost = fluid.layers.cross_entropy(input=out, label=label)

    avg_cost = fluid.layers.mean(x=cost)
    acc_top1 = fluid.layers.accuracy(input=out, label=label, k=1)
    acc_top5 = fluid.layers.accuracy(input=out, label=label, k=5)

    # for test
    inference_program = fluid.default_main_program().clone(for_test=True)

    # learning rate strategy
    epoch_points = [num_epochs / 3, num_epochs * 2 / 3]
    total_videos = args.total_videos
    step = int(total_videos / batch_size + 1)
    bd = [e * step for e in epoch_points]

    lr_init = args.lr_init
    lr = [lr_init, lr_init / 10, lr_init / 100]

    # initialize optimizer
    optimizer = fluid.optimizer.Momentum(
        learning_rate=fluid.layers.piecewise_decay(
            boundaries=bd, values=lr),
        momentum=0.9,
        regularization=fluid.regularizer.L2Decay(1e-4))

    opts = optimizer.minimize(avg_cost)
    if args.with_mem_opt:
        fluid.memory_optimize(fluid.default_main_program())

    place = fluid.CUDAPlace(0)
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    def is_parameter(var):
        if isinstance(var, Parameter):
            return isinstance(var, Parameter) and (not ("fc_0" in var.name))

    if pretrained_model is not None:
        vars = filter(is_parameter, inference_program.list_vars())
        fluid.io.load_vars(exe, pretrained_model, vars=vars)

    # reader
    train_reader = paddle.batch(reader.train(seg_num), batch_size=batch_size, drop_last=True)
    # test in single GPU
    test_reader = paddle.batch(reader.test(seg_num), batch_size=batch_size / 16)
    feeder = fluid.DataFeeder(place=place, feed_list=[image, label])

    train_exe = fluid.ParallelExecutor(use_cuda=True, loss_name=avg_cost.name)

    fetch_list = [avg_cost.name, acc_top1.name, acc_top5.name]

    # train
    for pass_id in range(num_epochs):
        train_info = [[], [], []]
        test_info = [[], [], []]
        for batch_id, data in enumerate(train_reader()):
            t1 = time.time()
            loss, acc1, acc5 = train_exe.run(fetch_list, feed=feeder.feed(data))
            t2 = time.time()
            period = t2 - t1
            loss = np.mean(np.array(loss))
            acc1 = np.mean(np.array(acc1))
            acc5 = np.mean(np.array(acc5))
            train_info[0].append(loss)
            train_info[1].append(acc1)
            train_info[2].append(acc5)

            if batch_id % 10 == 0:
                print(
                    "[TRAIN] Pass: {0}\ttrainbatch: {1}\tloss: {2}\tacc1: {3}\tacc5: {4}\ttime: {5}"
                    .format(pass_id, batch_id, '%.6f' % loss, acc1, acc5,
                            "%2.2f sec" % period))
                sys.stdout.flush()

        train_loss = np.array(train_info[0]).mean()
        train_acc1 = np.array(train_info[1]).mean()
        train_acc5 = np.array(train_info[2]).mean()

        # test
        cnt = 0
        for batch_id, data in enumerate(test_reader()):
            t1 = time.time()
            loss, acc1, acc5 = exe.run(inference_program,
                                       fetch_list=fetch_list,
                                       feed=feeder.feed(data))
            t2 = time.time()
            period = t2 - t1
            loss = np.mean(loss)
            acc1 = np.mean(acc1)
            acc5 = np.mean(acc5)
            test_info[0].append(loss * len(data))
            test_info[1].append(acc1 * len(data))
            test_info[2].append(acc5 * len(data))
            cnt += len(data)
            if batch_id % 10 == 0:
                print(
                    "[TEST] Pass: {0}\ttestbatch: {1}\tloss: {2}\tacc1: {3}\tacc5: {4}\ttime: {5}"
                    .format(pass_id, batch_id, '%.6f' % loss, acc1, acc5,
                            "%2.2f sec" % period))
                sys.stdout.flush()

        test_loss = np.sum(test_info[0]) / cnt
        test_acc1 = np.sum(test_info[1]) / cnt
        test_acc5 = np.sum(test_info[2]) / cnt

        print(
            "+ End pass: {0}, train_loss: {1}, train_acc1: {2}, train_acc5: {3}"
            .format(pass_id, '%.3f' % train_loss, '%.3f' % train_acc1, '%.3f' %
                    train_acc5))
        print("+ End pass: {0}, test_loss: {1}, test_acc1: {2}, test_acc5: {3}"
              .format(pass_id, '%.3f' % test_loss, '%.3f' % test_acc1, '%.3f' %
                      test_acc5))
        sys.stdout.flush()

        # save model
        model_path = os.path.join(model_save_dir, str(pass_id))
        if not os.path.isdir(model_path):
            os.makedirs(model_path)
        fluid.io.save_persistables(exe, model_path)


def main():
    args = parser.parse_args()
    print_arguments(args)
    train(args)


if __name__ == '__main__':
    main()
