#!/bin/env python

#function:
#   demo to show how to use converted model using caffe2fluid
#

import sys
import os
import numpy as np
import paddle.fluid as fluid
import paddle


def test_model(exe, test_program, fetch_list, test_reader, feeder):
    acc_set = []

    for data in test_reader():
        acc_np, pred = exe.run(program=test_program,
                               feed=feeder.feed(data),
                               fetch_list=fetch_list)
        acc_set.append(float(acc_np))

    acc_val = np.array(acc_set).mean()
    return float(acc_val)


def evaluate(net_file, model_file):
    """ main
    """
    #1, build model
    net_path = os.path.dirname(net_file)
    if net_path not in sys.path:
        sys.path.insert(0, net_path)

    from lenet import LeNet as MyNet

    #1, define network topology
    images = fluid.layers.data(name='image', shape=[1, 28, 28], dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')

    net = MyNet({'data': images})
    prediction = net.layers['prob']
    acc = fluid.layers.accuracy(input=prediction, label=label)

    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    #2, load weights
    if model_file.find('.npy') > 0:
        net.load(data_path=model_file, exe=exe, place=place)
    else:
        net.load(data_path=model_file, exe=exe)

    #3, test this model
    test_program = fluid.default_main_program().clone()
    test_reader = paddle.batch(paddle.dataset.mnist.test(), batch_size=128)

    feeder = fluid.DataFeeder(feed_list=[images, label], place=place)
    fetch_list = [acc, prediction]

    print('go to test model using test set')
    acc_val = test_model(exe, test_program, \
            fetch_list, test_reader, feeder)

    print('test accuracy is [%.4f], expected value[0.919]' % (acc_val))


if __name__ == "__main__":
    net_file = 'models/lenet/lenet.py'
    weight_file = 'models/lenet/lenet.npy'

    argc = len(sys.argv)
    if argc == 3:
        net_file = sys.argv[1]
        weight_file = sys.argv[2]
    elif argc > 1:
        print('usage:')
        print('\tpython %s [net_file] [weight_file]' % (sys.argv[0]))
        print('\teg:python %s %s %s %s' % (sys.argv[0], net_file, weight_file))
        sys.exit(1)

    evaluate(net_file, weight_file)
