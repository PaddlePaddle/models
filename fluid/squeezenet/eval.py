#coding:utf-8

import os
import sys
import numpy as np
import argparse
import functools
import paddle
import paddle.fluid as fluid
from utility import add_arguments, print_arguments
from squeezenet import squeeze_net
import reader

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('batch_size', int, 32, "Minibatch size.")
add_arg('use_gpu', bool, True, "Whether to use GPU or not.")
add_arg('test_list', str, 'data/ILSVRC2012/val_list.txt',
    "The testing data lists.")
add_arg('model_dir', str, './models/final', "The model path.")
# yapf: enable


def eval(args):
    class_dim = 1000
    image_shape = [3, 227, 227]
    image = fluid.layers.data(name='image', shape=image_shape, dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    out = squeeze_net(img=image, class_dim=class_dim)
    cost = fluid.layers.cross_entropy(input=out, label=label)
    acc_top1 = fluid.layers.accuracy(input=out, label=label, k=1)
    acc_top5 = fluid.layers.accuracy(input=out, label=label, k=5)
    avg_cost = fluid.layers.mean(x=cost)

    inference_program = fluid.default_main_program().clone(for_test=True)

    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)

    if not os.path.exists(args.model_dir):
        raise ValueError("The model path [%s] does not exist." %
                         (args.model_dir))
    if not os.path.exists(args.test_list):
        raise ValueError("The test lists [%s] does not exist." %
                         (args.test_list))

    def if_exist(var):
        return os.path.exists(os.path.join(args.model_dir, var.name))

    fluid.io.load_vars(exe, args.model_dir, predicate=if_exist)

    test_reader = paddle.batch(
        reader.test(file_list=args.test_list), batch_size=args.batch_size)
    feeder = fluid.DataFeeder(place=place, feed_list=[image, label])

    fetch_list = [avg_cost, acc_top1, acc_top5]

    test_info = [[], [], []]
    for batch_id, data in enumerate(test_reader()):
        loss, acc1, acc5 = exe.run(inference_program,
                                   feed=feeder.feed(data),
                                   fetch_list=fetch_list)
        test_info[0].append(loss[0])
        test_info[1].append(acc1[0])
        test_info[2].append(acc5[0])
        if batch_id % 1 == 0:
            print("Test {0}, loss {1}, acc1 {2}, acc5 {3}"
                  .format(batch_id, loss[0], acc1[0], acc5[0]))
            sys.stdout.flush()

    test_loss = np.array(test_info[0]).mean()
    test_acc1 = np.array(test_info[1]).mean()
    test_acc5 = np.array(test_info[2]).mean()

    print("Test loss {0}, acc1 {1}, acc5 {2}".format(test_loss, test_acc1,
                                                     test_acc5))
    sys.stdout.flush()


if __name__ == '__main__':
    args = parser.parse_args()
    print_arguments(args)
    eval(args)
