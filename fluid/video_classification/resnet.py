import os
import numpy as np
import time
import sys
import paddle
import paddle.fluid as fluid
import reader
import paddle.fluid.layers.control_flow as control_flow
import paddle.fluid.layers.nn as nn
import paddle.fluid.layers.tensor as tensor
import math


def conv_bn_layer(input, num_filters, filter_size, stride=1, groups=1,
                  act=None):
    conv = fluid.layers.conv2d(
        input=input,
        num_filters=num_filters,
        filter_size=filter_size,
        stride=stride,
        padding=(filter_size - 1) / 2,
        groups=groups,
        act=None,
        bias_attr=False)
    return fluid.layers.batch_norm(input=conv, act=act)


def shortcut(input, ch_out, stride):
    ch_in = input.shape[1]
    if ch_in != ch_out or stride != 1:
        return conv_bn_layer(input, ch_out, 1, stride)
    else:
        return input


def bottleneck_block(input, num_filters, stride):
    conv0 = conv_bn_layer(
        input=input, num_filters=num_filters, filter_size=1, act='relu')
    conv1 = conv_bn_layer(
        input=conv0,
        num_filters=num_filters,
        filter_size=3,
        stride=stride,
        act='relu')
    conv2 = conv_bn_layer(
        input=conv1, num_filters=num_filters * 4, filter_size=1, act=None)

    short = shortcut(input, num_filters * 4, stride)

    return fluid.layers.elementwise_add(x=short, y=conv2, act='relu')


def ResNet(input, seg_num, class_dim, layers=50):
    supported_layers = [50, 101, 152]
    if layers not in supported_layers:
        print("supported layers are", supported_layers, \
              "but input layer is ", layers)
        exit()

    # reshape input
    input = fluid.layers.reshape(x=input, shape=[-1, 3, 224, 224])

    if layers == 50:
        depth = [3, 4, 6, 3]
    elif layers == 101:
        depth = [3, 4, 23, 3]
    elif layers == 152:
        depth = [3, 8, 36, 3]
    num_filters = [64, 128, 256, 512]

    conv = conv_bn_layer(
        input=input, num_filters=64, filter_size=7, stride=2, act='relu')
    conv = fluid.layers.pool2d(
        input=conv, pool_size=3, pool_stride=2, pool_padding=1, pool_type='max')

    for block in range(len(depth)):
        for i in range(depth[block]):
            conv = bottleneck_block(
                input=conv,
                num_filters=num_filters[block],
                stride=2 if i == 0 and block != 0 else 1)
    pool = fluid.layers.pool2d(
        input=conv, pool_size=7, pool_type='avg', global_pooling=True)

    feature = fluid.layers.reshape(x=pool, shape=[-1, seg_num, pool.shape[1]])
    out = fluid.layers.reduce_mean(feature, dim=1)

    stdv = 1.0 / math.sqrt(pool.shape[1] * 1.0)
    out = fluid.layers.fc(input=out,
                          size=class_dim,
                          act='softmax',
                          param_attr=fluid.param_attr.ParamAttr(
                              initializer=fluid.initializer.Uniform(-stdv,
                                                                    stdv)))
    return out
