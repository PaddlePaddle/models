import os

import paddle as paddle
import paddle.fluid as fluid
from paddle.fluid.initializer import Xavier
from paddle.fluid.initializer import Normal
from paddle.fluid.initializer import Constant
from paddle.fluid.param_attr import ParamAttr


def conv_bn_layer(input,
                  name,
                  filter_size,
                  num_filters,
                  stride,
                  padding=0,
                  channels=None,
                  num_groups=1,
                  act='relu',
                  use_bn=True,
                  use_cudnn=True):
    parameter_attr = ParamAttr(name=name + '_weights', initializer=Xavier())
    parameter_attr_bias = ParamAttr(
        name=name + '_biases', initializer=Constant())
    conv = fluid.layers.conv2d(
        input=input,
        name=name,
        num_filters=num_filters,
        filter_size=filter_size,
        stride=stride,
        padding=padding,
        groups=num_groups,
        act=None if use_bn else act,
        use_cudnn=use_cudnn,
        param_attr=parameter_attr,
        bias_attr=parameter_attr_bias)
    return fluid.layers.batch_norm(input=conv, act=act) if use_bn else conv


def fire_module(input,
                name,
                num_filters_s1,
                num_filters_e1,
                num_filters_e3,
                use_bn=True):
    """
    """
    s1 = conv_bn_layer(
        input=input,
        name=name + '_squeeze1x1',
        filter_size=1,
        num_filters=num_filters_s1,
        stride=1,
        padding=0,
        use_bn=use_bn)

    e1 = conv_bn_layer(
        input=s1,
        name=name + '_expand1x1',
        filter_size=1,
        num_filters=num_filters_e1,
        stride=1,
        padding=0,
        use_bn=use_bn)

    e3 = conv_bn_layer(
        input=s1,
        name=name + '_expand3x3',
        filter_size=3,
        num_filters=num_filters_e3,
        stride=1,
        padding=1,
        use_bn=use_bn)
    return fluid.layers.concat(input=[e1, e3], axis=1)


def squeeze_net(img, class_dim, use_bn=True):
    # conv1: 113x113
    conv1 = conv_bn_layer(
        img,
        name='conv1',
        channels=3,
        filter_size=3,
        num_filters=64,
        stride=2,
        use_bn=use_bn)
    #
    pool1 = fluid.layers.pool2d(
        input=conv1,
        name='pool1',
        pool_size=3,
        pool_stride=2,
        pool_type='max',
        pool_padding=0)

    fire2 = fire_module(
        input=pool1,
        name='fire2',
        num_filters_s1=16,
        num_filters_e1=64,
        num_filters_e3=64,
        use_bn=use_bn)

    fire3 = fire_module(
        input=fire2,
        name='fire3',
        num_filters_s1=16,
        num_filters_e1=64,
        num_filters_e3=64,
        use_bn=use_bn)

    pool3 = fluid.layers.pool2d(
        input=fire3,
        name='pool3',
        pool_size=3,
        pool_stride=2,
        pool_type='max',
        pool_padding=0)

    fire4 = fire_module(
        input=pool3,
        name='fire4',
        num_filters_s1=32,
        num_filters_e1=128,
        num_filters_e3=128,
        use_bn=use_bn)

    fire5 = fire_module(
        input=fire4,
        name='fire5',
        num_filters_s1=32,
        num_filters_e1=128,
        num_filters_e3=128,
        use_bn=use_bn)

    pool5 = fluid.layers.pool2d(
        input=fire5,
        name='pool5',
        pool_size=3,
        pool_stride=2,
        pool_type='max',
        pool_padding=0)

    fire6 = fire_module(
        input=pool5,
        name='fire6',
        num_filters_s1=48,
        num_filters_e1=192,
        num_filters_e3=192,
        use_bn=use_bn)

    fire7 = fire_module(
        input=fire6,
        name='fire7',
        num_filters_s1=48,
        num_filters_e1=192,
        num_filters_e3=192,
        use_bn=use_bn)

    fire8 = fire_module(
        input=fire7,
        name='fire8',
        num_filters_s1=64,
        num_filters_e1=256,
        num_filters_e3=256,
        use_bn=use_bn)

    fire9 = fire_module(
        input=fire8,
        name='fire9',
        num_filters_s1=64,
        num_filters_e1=256,
        num_filters_e3=256,
        use_bn=use_bn)

    drop9 = fluid.layers.dropout(x=fire9, dropout_prob=0.5)

    conv10 = conv_bn_layer(
        input=drop9,
        name='conv10',
        filter_size=1,
        num_filters=class_dim,
        stride=1,
        use_bn=use_bn)

    pool10 = fluid.layers.pool2d(
        input=conv10,
        name='pool10',
        pool_size=0,
        pool_stride=1,
        pool_type='avg',
        global_pooling=True)

    pool10 = fluid.layers.reshape(
        x=pool10, shape=[-1, class_dim], act='softmax', inplace=True)
    return pool10
