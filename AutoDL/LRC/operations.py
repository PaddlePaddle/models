#  Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.
#
# Based on:
# --------------------------------------------------------
# DARTS
# Copyright (c) 2018, Hanxiao Liu.
# Licensed under the Apache License, Version 2.0;
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import numpy as np
import time
import paddle
import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.initializer import Xavier
from paddle.fluid.initializer import Normal
from paddle.fluid.initializer import Constant

OPS = {
    'none' : lambda input, C, stride, name, affine: Zero(input, stride, name),
    'avg_pool_3x3' : lambda input, C, stride, name, affine: fluid.layers.pool2d(input, 3, 'avg', pool_stride=stride, pool_padding=1, name=name),
    'max_pool_3x3' : lambda input, C, stride, name, affine: fluid.layers.pool2d(input, 3, 'max', pool_stride=stride, pool_padding=1, name=name),
    'skip_connect' : lambda input,C, stride, name, affine: Identity(input, name) if stride == 1 else FactorizedReduce(input, C, name=name, affine=affine),
    'sep_conv_3x3' : lambda input,C, stride, name, affine: SepConv(input, C, C, 3, stride, 1, name=name, affine=affine),
    'sep_conv_5x5' : lambda input,C, stride, name, affine: SepConv(input, C, C, 5, stride, 2, name=name, affine=affine),
    'sep_conv_7x7' : lambda input,C, stride, name, affine: SepConv(input, C, C, 7, stride, 3, name=name, affine=affine),
    'dil_conv_3x3' : lambda input,C, stride, name, affine: DilConv(input, C, C, 3, stride, 2, 2, name=name, affine=affine),
    'dil_conv_5x5' : lambda input,C, stride, name, affine: DilConv(input, C, C, 5, stride, 4, 2, name=name, affine=affine),
    'conv_7x1_1x7' : lambda input,C, stride, name, affine: SevenConv(input, C, name=name, affine=affine)
}


def ReLUConvBN(input, C_out, kernel_size, stride, padding, name='',
               affine=True):
    relu_a = fluid.layers.relu(input)
    conv2d_a = fluid.layers.conv2d(
        relu_a,
        C_out,
        kernel_size,
        stride,
        padding,
        param_attr=ParamAttr(
            initializer=Xavier(
                uniform=False, fan_in=0),
            name=name + 'op.1.weight'),
        bias_attr=False)
    if affine:
        reluconvbn_out = fluid.layers.batch_norm(
            conv2d_a,
            param_attr=ParamAttr(
                initializer=Constant(1.), name=name + 'op.2.weight'),
            bias_attr=ParamAttr(
                initializer=Constant(0.), name=name + 'op.2.bias'),
            moving_mean_name=name + 'op.2.running_mean',
            moving_variance_name=name + 'op.2.running_var')
    else:
        reluconvbn_out = fluid.layers.batch_norm(
            conv2d_a,
            param_attr=ParamAttr(
                initializer=Constant(1.),
                learning_rate=0.,
                name=name + 'op.2.weight'),
            bias_attr=ParamAttr(
                initializer=Constant(0.),
                learning_rate=0.,
                name=name + 'op.2.bias'),
            moving_mean_name=name + 'op.2.running_mean',
            moving_variance_name=name + 'op.2.running_var')
    return reluconvbn_out


def DilConv(input,
            C_in,
            C_out,
            kernel_size,
            stride,
            padding,
            dilation,
            name='',
            affine=True):
    relu_a = fluid.layers.relu(input)
    conv2d_a = fluid.layers.conv2d(
        relu_a,
        C_in,
        kernel_size,
        stride,
        padding,
        dilation,
        groups=C_in,
        param_attr=ParamAttr(
            initializer=Xavier(
                uniform=False, fan_in=0),
            name=name + 'op.1.weight'),
        bias_attr=False,
        use_cudnn=False)
    conv2d_b = fluid.layers.conv2d(
        conv2d_a,
        C_out,
        1,
        param_attr=ParamAttr(
            initializer=Xavier(
                uniform=False, fan_in=0),
            name=name + 'op.2.weight'),
        bias_attr=False)
    if affine:
        dilconv_out = fluid.layers.batch_norm(
            conv2d_b,
            param_attr=ParamAttr(
                initializer=Constant(1.), name=name + 'op.3.weight'),
            bias_attr=ParamAttr(
                initializer=Constant(0.), name=name + 'op.3.bias'),
            moving_mean_name=name + 'op.3.running_mean',
            moving_variance_name=name + 'op.3.running_var')
    else:
        dilconv_out = fluid.layers.batch_norm(
            conv2d_b,
            param_attr=ParamAttr(
                initializer=Constant(1.),
                learning_rate=0.,
                name=name + 'op.3.weight'),
            bias_attr=ParamAttr(
                initializer=Constant(0.),
                learning_rate=0.,
                name=name + 'op.3.bias'),
            moving_mean_name=name + 'op.3.running_mean',
            moving_variance_name=name + 'op.3.running_var')
    return dilconv_out


def SepConv(input,
            C_in,
            C_out,
            kernel_size,
            stride,
            padding,
            name='',
            affine=True):
    relu_a = fluid.layers.relu(input)
    conv2d_a = fluid.layers.conv2d(
        relu_a,
        C_in,
        kernel_size,
        stride,
        padding,
        groups=C_in,
        param_attr=ParamAttr(
            initializer=Xavier(
                uniform=False, fan_in=0),
            name=name + 'op.1.weight'),
        bias_attr=False,
        use_cudnn=False)
    conv2d_b = fluid.layers.conv2d(
        conv2d_a,
        C_in,
        1,
        param_attr=ParamAttr(
            initializer=Xavier(
                uniform=False, fan_in=0),
            name=name + 'op.2.weight'),
        bias_attr=False)
    if affine:
        bn_a = fluid.layers.batch_norm(
            conv2d_b,
            param_attr=ParamAttr(
                initializer=Constant(1.), name=name + 'op.3.weight'),
            bias_attr=ParamAttr(
                initializer=Constant(0.), name=name + 'op.3.bias'),
            moving_mean_name=name + 'op.3.running_mean',
            moving_variance_name=name + 'op.3.running_var')
    else:
        bn_a = fluid.layers.batch_norm(
            conv2d_b,
            param_attr=ParamAttr(
                initializer=Constant(1.),
                learning_rate=0.,
                name=name + 'op.3.weight'),
            bias_attr=ParamAttr(
                initializer=Constant(0.),
                learning_rate=0.,
                name=name + 'op.3.bias'),
            moving_mean_name=name + 'op.3.running_mean',
            moving_variance_name=name + 'op.3.running_var')

    relu_b = fluid.layers.relu(bn_a)
    conv2d_d = fluid.layers.conv2d(
        relu_b,
        C_in,
        kernel_size,
        1,
        padding,
        groups=C_in,
        param_attr=ParamAttr(
            initializer=Xavier(
                uniform=False, fan_in=0),
            name=name + 'op.5.weight'),
        bias_attr=False,
        use_cudnn=False)
    conv2d_e = fluid.layers.conv2d(
        conv2d_d,
        C_out,
        1,
        param_attr=ParamAttr(
            initializer=Xavier(
                uniform=False, fan_in=0),
            name=name + 'op.6.weight'),
        bias_attr=False)
    if affine:
        sepconv_out = fluid.layers.batch_norm(
            conv2d_e,
            param_attr=ParamAttr(
                initializer=Constant(1.), name=name + 'op.7.weight'),
            bias_attr=ParamAttr(
                initializer=Constant(0.), name=name + 'op.7.bias'),
            moving_mean_name=name + 'op.7.running_mean',
            moving_variance_name=name + 'op.7.running_var')
    else:
        sepconv_out = fluid.layers.batch_norm(
            conv2d_e,
            param_attr=ParamAttr(
                initializer=Constant(1.),
                learning_rate=0.,
                name=name + 'op.7.weight'),
            bias_attr=ParamAttr(
                initializer=Constant(0.),
                learning_rate=0.,
                name=name + 'op.7.bias'),
            moving_mean_name=name + 'op.7.running_mean',
            moving_variance_name=name + 'op.7.running_var')
    return sepconv_out


def SevenConv(input, C_out, stride, name='', affine=True):
    relu_a = fluid.layers.relu(input)
    conv2d_a = fluid.layers.conv2d(
        relu_a,
        C_out, (1, 7), (1, stride), (0, 3),
        param_attr=ParamAttr(
            initializer=Xavier(
                uniform=False, fan_in=0),
            name=name + 'op.1.weight'),
        bias_attr=False)
    conv2d_b = fluid.layers.conv2d(
        conv2d_a,
        C_out, (7, 1), (stride, 1), (3, 0),
        param_attr=ParamAttr(
            initializer=Xavier(
                uniform=False, fan_in=0),
            name=name + 'op.2.weight'),
        bias_attr=False)
    if affine:
        out = fluid.layers.batch_norm(
            conv2d_b,
            param_attr=ParamAttr(
                initializer=Constant(1.), name=name + 'op.3.weight'),
            bias_attr=ParamAttr(
                initializer=Constant(0.), name=name + 'op.3.bias'),
            moving_mean_name=name + 'op.3.running_mean',
            moving_variance_name=name + 'op.3.running_var')
    else:
        out = fluid.layers.batch_norm(
            conv2d_b,
            param_attr=ParamAttr(
                initializer=Constant(1.),
                learning_rate=0.,
                name=name + 'op.3.weight'),
            bias_attr=ParamAttr(
                initializer=Constant(0.),
                learning_rate=0.,
                name=name + 'op.3.bias'),
            moving_mean_name=name + 'op.3.running_mean',
            moving_variance_name=name + 'op.3.running_var')


def Identity(input, name=''):
    return input


def Zero(input, stride, name=''):
    ones = np.ones(input.shape[-2:])
    ones[::stride, ::stride] = 0
    ones = fluid.layers.assign(ones)
    return input * ones


def FactorizedReduce(input, C_out, name='', affine=True):
    relu_a = fluid.layers.relu(input)
    conv2d_a = fluid.layers.conv2d(
        relu_a,
        C_out // 2,
        1,
        2,
        param_attr=ParamAttr(
            initializer=Xavier(
                uniform=False, fan_in=0),
            name=name + 'conv_1.weight'),
        bias_attr=False)
    h_end = relu_a.shape[2]
    w_end = relu_a.shape[3]
    slice_a = fluid.layers.slice(relu_a, [2, 3], [1, 1], [h_end, w_end])
    conv2d_b = fluid.layers.conv2d(
        slice_a,
        C_out // 2,
        1,
        2,
        param_attr=ParamAttr(
            initializer=Xavier(
                uniform=False, fan_in=0),
            name=name + 'conv_2.weight'),
        bias_attr=False)
    out = fluid.layers.concat([conv2d_a, conv2d_b], axis=1)
    if affine:
        out = fluid.layers.batch_norm(
            out,
            param_attr=ParamAttr(
                initializer=Constant(1.), name=name + 'bn.weight'),
            bias_attr=ParamAttr(
                initializer=Constant(0.), name=name + 'bn.bias'),
            moving_mean_name=name + 'bn.running_mean',
            moving_variance_name=name + 'bn.running_var')
    else:
        out = fluid.layers.batch_norm(
            out,
            param_attr=ParamAttr(
                initializer=Constant(1.),
                learning_rate=0.,
                name=name + 'bn.weight'),
            bias_attr=ParamAttr(
                initializer=Constant(0.),
                learning_rate=0.,
                name=name + 'bn.bias'),
            moving_mean_name=name + 'bn.running_mean',
            moving_variance_name=name + 'bn.running_var')
    return out
