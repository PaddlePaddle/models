#  Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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
import functools
import paddle
import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.initializer import Xavier
from paddle.fluid.initializer import Normal
from paddle.fluid.initializer import Constant


from collections import namedtuple
Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

arch_dict = {
    'DARTS_6M': Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 1), ('skip_connect', 4), ('sep_conv_3x3', 3)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 0), ('max_pool_3x3', 1), ('dil_conv_5x5', 2), ('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('dil_conv_5x5', 3), ('dil_conv_3x3', 1), ('sep_conv_3x3', 2)], reduce_concat=range(2, 6)), 
    'DARTS_4M': Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 0), ('dil_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('avg_pool_3x3', 1), ('skip_connect', 3), ('skip_connect', 2), ('sep_conv_3x3', 0), ('sep_conv_5x5', 2)], reduce_concat=range(2, 6)),
}

__all__ = list(arch_dict.keys())

train_parameters = {
    "input_size": [3, 224, 224],
    "input_mean": [0.485, 0.456, 0.406],
    "input_std": [0.229, 0.224, 0.225],
    "learning_strategy": {
        "name": "piecewise_decay",
        "batch_size": 256,
        "epochs": [30, 60, 90],
        "steps": [0.1, 0.01, 0.001, 0.0001]
    }
}

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
        bias_attr=False,
        use_cudnn=False)
    conv2d_b = fluid.layers.conv2d(
        conv2d_a,
        C_out,
        1,
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
        bias_attr=False,
        use_cudnn=False)
    conv2d_b = fluid.layers.conv2d(
        conv2d_a,
        C_in,
        1,
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
        bias_attr=False,
        use_cudnn=False)
    conv2d_e = fluid.layers.conv2d(
        conv2d_d,
        C_out,
        1,
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

class Cell():
    def __init__(self, genotype, C_prev_prev, C_prev, C, reduction,
                 reduction_prev):

        if reduction_prev:
            self.preprocess0 = functools.partial(FactorizedReduce, C_out=C)
        else:
            self.preprocess0 = functools.partial(
                ReLUConvBN, C_out=C, kernel_size=1, stride=1, padding=0)
        self.preprocess1 = functools.partial(
            ReLUConvBN, C_out=C, kernel_size=1, stride=1, padding=0)
        if reduction:
            op_names, indices = zip(*genotype.reduce)
            concat = genotype.reduce_concat
        else:
            op_names, indices = zip(*genotype.normal)
            concat = genotype.normal_concat
        print(op_names, indices, concat, reduction)
        self._compile(C, op_names, indices, concat, reduction)

    def _compile(self, C, op_names, indices, concat, reduction):
        assert len(op_names) == len(indices)
        self._steps = len(op_names) // 2
        self._concat = concat
        self.multiplier = len(concat)

        self._ops = []
        for name, index in zip(op_names, indices):
            stride = 2 if reduction and index < 2 else 1
            op = functools.partial(OPS[name], C=C, stride=stride, affine=True)
            self._ops += [op]
        self._indices = indices

    def forward(self, s0, s1, drop_prob, is_train, name):
        self.training = is_train
        preprocess0_name = name + 'preprocess0.'
        preprocess1_name = name + 'preprocess1.'
        s0 = self.preprocess0(s0, name=preprocess0_name)
        s1 = self.preprocess1(s1, name=preprocess1_name)
        out = [s0, s1]
        for i in range(self._steps):
            h1 = out[self._indices[2 * i]]
            h2 = out[self._indices[2 * i + 1]]
            op1 = self._ops[2 * i]
            op2 = self._ops[2 * i + 1]
            h3 = op1(h1, name=name + '_ops.' + str(2 * i) + '.')
            h4 = op2(h2, name=name + '_ops.' + str(2 * i + 1) + '.')
            if self.training and drop_prob > 0.:
                if h3 != h1:
                    h3 = fluid.layers.dropout(
                        h3,
                        drop_prob,
                        dropout_implementation='upscale_in_train')
                if h4 != h2:
                    h4 = fluid.layers.dropout(
                        h4,
                        drop_prob,
                        dropout_implementation='upscale_in_train')
            s = h3 + h4
            out += [s]
        return fluid.layers.concat([out[i] for i in self._concat], axis=1)

def AuxiliaryHeadImageNet(input, num_classes, aux_name='auxiliary_head'):
    relu_a = fluid.layers.relu(input)
    pool_a = fluid.layers.pool2d(relu_a, 5, 'avg', 2)
    conv2d_a = fluid.layers.conv2d(
        pool_a,
        128,
        1,
        name=aux_name + '.features.2',
        bias_attr=False)
    bn_a_name = aux_name + '.features.3'
    bn_a = fluid.layers.batch_norm(
        conv2d_a,
        act='relu',
        name=bn_a_name,
        param_attr=ParamAttr(
            initializer=Constant(1.), name=bn_a_name + '.weight'),
        bias_attr=ParamAttr(
            initializer=Constant(0.), name=bn_a_name + '.bias'),
        moving_mean_name=bn_a_name + '.running_mean',
        moving_variance_name=bn_a_name + '.running_var')
    conv2d_b = fluid.layers.conv2d(
        bn_a,
        768,
        2,
        name=aux_name + '.features.5',
        bias_attr=False)
    bn_b_name = aux_name + '.features.6'
    bn_b = fluid.layers.batch_norm(
        conv2d_b,
        act='relu',
        name=bn_b_name,
        param_attr=ParamAttr(
            initializer=Constant(1.), name=bn_b_name + '.weight'),
        bias_attr=ParamAttr(
            initializer=Constant(0.), name=bn_b_name + '.bias'),
        moving_mean_name=bn_b_name + '.running_mean',
        moving_variance_name=bn_b_name + '.running_var')
    pool_b = fluid.layers.adaptive_pool2d(bn_b, (1, 1), "avg") 
    fc_name = aux_name + '.classifier'
    fc = fluid.layers.fc(pool_b,
                         num_classes,
                         name=fc_name,
                         param_attr=ParamAttr(
                             initializer=Normal(scale=1e-3),
                             name=fc_name + '.weight'),
                         bias_attr=ParamAttr(
                             initializer=Constant(0.), name=fc_name + '.bias'))
    return fc


def StemConv0(input, C_out):
    conv_a = fluid.layers.conv2d(
        input,
        C_out // 2,
        3,
        stride=2,
        padding=1,
        bias_attr=False)
    bn_a = fluid.layers.batch_norm(
        conv_a,
        act='relu',
        param_attr=ParamAttr(
            initializer=Constant(1.), name='stem0.1.weight'),
        bias_attr=ParamAttr(
            initializer=Constant(0.), name='stem0.1.bias'),
        moving_mean_name='stem0.1.running_mean',
        moving_variance_name='stem0.1.running_var')
    
    conv_b = fluid.layers.conv2d(
        bn_a,
        C_out,
        3,
        stride=2,
        padding=1,
        bias_attr=False)
    bn_b = fluid.layers.batch_norm(
        conv_b,
        param_attr=ParamAttr(
            initializer=Constant(1.), name='stem0.3.weight'),
        bias_attr=ParamAttr(
            initializer=Constant(0.), name='stem0.3.bias'),
        moving_mean_name='stem0.3.running_mean',
        moving_variance_name='stem0.3.running_var')
    return bn_b

def StemConv1(input, C_out):
    relu_a = fluid.layers.relu(input)
    conv_a = fluid.layers.conv2d(
        relu_a,
        C_out,
        3,
        stride=2,
        padding=1,
        bias_attr=False)
    bn_a = fluid.layers.batch_norm(
        conv_a,
        param_attr=ParamAttr(
            initializer=Constant(1.), name='stem1.1.weight'),
        bias_attr=ParamAttr(
            initializer=Constant(0.), name='stem1.1.bias'),
        moving_mean_name='stem1.1.running_mean',
        moving_variance_name='stem1.1.running_var')
    return bn_a

class NetworkImageNet(object):
    def __init__(self, arch='DARTS_6M'):
        self.params = train_parameters
        self.class_num = 1000
        self.init_channel = 48
        self._layers = 14
        self._auxiliary = False
        self.drop_path_prob = 0
        genotype = arch_dict[arch]
        
        C = self.init_channel
        layers = self._layers
        C_prev_prev, C_prev, C_curr = C, C, C
        self.cells = []
        reduction_prev = True
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction,
                        reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
            if i == 2 * layers // 3:
                C_to_auxiliary = C_prev

    def net(self, input, class_dim=1000, is_train=True):
        self.logits_aux = None
        num_channel = self.init_channel
        s0 = StemConv0(input, num_channel)
        s1 = StemConv1(s0, num_channel)
        for i, cell in enumerate(self.cells):
            name = 'cells.' + str(i) + '.'
            s0, s1 = s1, cell.forward(s0, s1, self.drop_path_prob, is_train,
                                      name)
            if i == int(2 * self._layers // 3):
                if self._auxiliary and is_train:
                    self.logits_aux = AuxiliaryHeadImageNet(s1, self.class_num)
        out = fluid.layers.adaptive_pool2d(s1, (1, 1), "avg")
        self.logits = fluid.layers.fc(out,
                                      size=self.class_num,
                                      param_attr=ParamAttr(
                                          initializer=Normal(scale=1e-4),
                                          name='classifier.weight'),
                                      bias_attr=ParamAttr(
                                          initializer=Constant(0.),
                                          name='classifier.bias'))
        return self.logits

def DARTS_6M():
    return NetworkImageNet(arch = 'DARTS_6M')
def DARTS_4M():
    return NetworkImageNet(arch = 'DARTS_4M')
