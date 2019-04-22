from __future__ import print_function

import contextlib
import unittest
import numpy as np
import six

import paddle
import paddle.fluid as fluid
from paddle.fluid import core
from paddle.fluid.optimizer import SGDOptimizer
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, FC
from paddle.fluid.dygraph.base import to_variable
import time


class SimpleImgConvPool(fluid.dygraph.Layer):
    """
    Conv Pool Layer
    """
    def __init__(self,
                 name_scope,
                 num_channels,
                 num_filters,
                 filter_size,
                 pool_size,
                 pool_stride,
                 pool_padding=0,
                 pool_type='max',
                 global_pooling=False,
                 conv_stride=1,
                 conv_padding=0,
                 conv_dilation=1,
                 conv_groups=1,
                 act=None,
                 use_cudnn=False,
                 param_attr=None,
                 bias_attr=None):
        super(SimpleImgConvPool, self).__init__(name_scope)

        self._conv2d = Conv2D(
            self.full_name(),
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=conv_stride,
            padding=conv_padding,
            dilation=conv_dilation,
            groups=conv_groups,
            param_attr=None,
            bias_attr=None,
            use_cudnn=use_cudnn)

        self._pool2d = Pool2D(
            self.full_name(),
            pool_size=pool_size,
            pool_type=pool_type,
            pool_stride=pool_stride,
            pool_padding=pool_padding,
            global_pooling=global_pooling,
            use_cudnn=use_cudnn)

    def forward(self, inputs):
        x = self._conv2d(inputs)
        x = self._pool2d(x)
        return x


class MNIST(fluid.dygraph.Layer):
    """
    MNIST model
    """
    def __init__(self, name_scope):
        super(MNIST, self).__init__(name_scope)

        self._simple_img_conv_pool_1 = SimpleImgConvPool(
            self.full_name(), 1, 20, 5, 2, 2, act="relu")

        self._simple_img_conv_pool_2 = SimpleImgConvPool(
            self.full_name(), 20, 50, 5, 2, 2, act="relu")

        pool_2_shape = 50 * 4 * 4
        SIZE = 10
        scale = (2.0 / (pool_2_shape**2 * SIZE))**0.5
        self._fc = FC(self.full_name(),
                      10,
                      param_attr=fluid.param_attr.ParamAttr(
                          initializer=fluid.initializer.NormalInitializer(
                              loc=0.0, scale=scale)),
                      act="softmax")

    def forward(self, inputs):
        x = self._simple_img_conv_pool_1(inputs)
        x = self._simple_img_conv_pool_2(x)
        x = self._fc(x)
        return x


def train_mnist():
    seed = 90
    epoch_num = 10

    with fluid.dygraph.guard():
        fluid.default_startup_program().random_seed = seed
        fluid.default_main_program().random_seed = seed

        mnist = MNIST("mnist")
        sgd = SGDOptimizer(learning_rate=1e-3)
        train_reader = paddle.batch(
            paddle.dataset.mnist.train(), batch_size=128, drop_last=True)

        for epoch in range(epoch_num):
            for batch_id, data in enumerate(train_reader()):
                dy_x_data = np.array(
                    [x[0].reshape(1, 28, 28)
                     for x in data]).astype('float32')
                y_data = np.array(
                    [x[1] for x in data]).astype('int64').reshape(128, 1)

                img = to_variable(dy_x_data)
                label = to_variable(y_data)
                label.stop_gradient = True

                cost = mnist(img)
                loss = fluid.layers.cross_entropy(cost, label)
                avg_loss = fluid.layers.mean(loss)
                avg_loss.backward()
                sgd.minimize(avg_loss)
                mnist.clear_gradients()

                dy_out = avg_loss.numpy()
                print("batch id %d, loss %f" % (batch_id, dy_out))


if __name__ == '__main__':
    train_mnist()

