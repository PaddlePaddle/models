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


def squeeze_excitation(input, num_channels, reduction_ratio):
    pool = fluid.layers.pool2d(
        input=input, pool_size=0, pool_type='avg', global_pooling=True)
    stdv = 1.0 / math.sqrt(pool.shape[1] * 1.0)
    squeeze = fluid.layers.fc(input=pool,
                              size=num_channels / reduction_ratio,
                              act='relu',
                              param_attr=fluid.param_attr.ParamAttr(
                                  initializer=fluid.initializer.Uniform(-stdv,
                                                                        stdv)))
    stdv = 1.0 / math.sqrt(squeeze.shape[1] * 1.0)
    excitation = fluid.layers.fc(input=squeeze,
                                 size=num_channels,
                                 act='sigmoid',
                                 param_attr=fluid.param_attr.ParamAttr(
                                     initializer=fluid.initializer.Uniform(
                                         -stdv, stdv)))
    scale = fluid.layers.elementwise_mul(x=input, y=excitation, axis=0)
    return scale


def shortcut(input, ch_out, stride):
    ch_in = input.shape[1]
    if ch_in != ch_out or stride != 1:
        filter_size = 1
        return conv_bn_layer(input, ch_out, filter_size, stride)
    else:
        return input


def bottleneck_block(input, num_filters, stride, cardinality, reduction_ratio):
    conv0 = conv_bn_layer(
        input=input, num_filters=num_filters, filter_size=1, act='relu')
    conv1 = conv_bn_layer(
        input=conv0,
        num_filters=num_filters,
        filter_size=3,
        stride=stride,
        groups=cardinality,
        act='relu')
    conv2 = conv_bn_layer(
        input=conv1, num_filters=num_filters * 2, filter_size=1, act=None)
    scale = squeeze_excitation(
        input=conv2,
        num_channels=num_filters * 2,
        reduction_ratio=reduction_ratio)

    short = shortcut(input, num_filters * 2, stride)

    return fluid.layers.elementwise_add(x=short, y=scale, act='relu')


def SE_ResNeXt(input, class_dim, infer=False, layers=50):
    supported_layers = [50, 152]
    if layers not in supported_layers:
        print("supported layers are", supported_layers, \
              "but input layer is ", layers)
        exit()
    if layers == 50:
        cardinality = 32
        reduction_ratio = 16
        depth = [3, 4, 6, 3]
        num_filters = [128, 256, 512, 1024]

        conv = conv_bn_layer(
            input=input, num_filters=64, filter_size=7, stride=2, act='relu')
        conv = fluid.layers.pool2d(
            input=conv,
            pool_size=3,
            pool_stride=2,
            pool_padding=1,
            pool_type='max')
    elif layers == 152:
        cardinality = 64
        reduction_ratio = 16
        depth = [3, 8, 36, 3]
        num_filters = [128, 256, 512, 1024]

        conv = conv_bn_layer(
            input=input, num_filters=64, filter_size=3, stride=2, act='relu')
        conv = conv_bn_layer(
            input=conv, num_filters=64, filter_size=3, stride=1, act='relu')
        conv = conv_bn_layer(
            input=conv, num_filters=128, filter_size=3, stride=1, act='relu')
        conv = fluid.layers.pool2d(
            input=conv, pool_size=3, pool_stride=2, pool_padding=1, \
            pool_type='max')

    for block in range(len(depth)):
        for i in range(depth[block]):
            conv = bottleneck_block(
                input=conv,
                num_filters=num_filters[block],
                stride=2 if i == 0 and block != 0 else 1,
                cardinality=cardinality,
                reduction_ratio=reduction_ratio)

    pool = fluid.layers.pool2d(
        input=conv, pool_size=7, pool_type='avg', global_pooling=True)
    if not infer:
        drop = fluid.layers.dropout(x=pool, dropout_prob=0.5)
    else:
        drop = pool
    stdv = 1.0 / math.sqrt(drop.shape[1] * 1.0)
    out = fluid.layers.fc(input=drop,
                          size=class_dim,
                          act='softmax',
                          param_attr=fluid.param_attr.ParamAttr(
                              initializer=fluid.initializer.Uniform(-stdv,
                                                                    stdv)))
    return out
