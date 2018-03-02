#!/bin/env python

#function:
#   demo to show how to use converted model using caffe2fluid
#

import numpy as np
import paddle.v2 as paddle
import paddle.v2.fluid as fluid

from lenet import LeNet as MyNet


def test_model(exe, test_program, fetch_list, test_reader, feeder):
    acc_set = []

    for data in test_reader():
        acc_np, pred = exe.run(program=test_program,
                               feed=feeder.feed(data),
                               fetch_list=fetch_list)
        acc_set.append(float(acc_np))

    acc_val = np.array(acc_set).mean()
    return float(acc_val)


def main(model_path):
    """ main
    """
    print('load fluid model in %s' % (model_path))

    with_gpu = False
    paddle.init(use_gpu=with_gpu)

    #1, define network topology
    images = fluid.layers.data(name='image', shape=[1, 28, 28], dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')

    net = MyNet({'data': images})
    prediction = net.layers['prob']
    acc = fluid.layers.accuracy(input=prediction, label=label)

    place = fluid.CUDAPlace(0) if with_gpu is True else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    #2, load weights
    if model_path.find('.npy') > 0:
        net.load(data_path=model_path, exe=exe, place=place)
    else:
        net.load(data_path=model_path, exe=exe)

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
    import sys
    if len(sys.argv) == 2:
        fluid_model_path = sys.argv[1]
    else:
        fluid_model_path = './model.fluid'

    main(fluid_model_path)
