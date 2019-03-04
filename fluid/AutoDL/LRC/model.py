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
import functools
import paddle
import paddle.fluid as fluid
from operations import *


class Cell():
    def __init__(self, genotype, C_prev_prev, C_prev, C, reduction,
                 reduction_prev):
        print(C_prev_prev, C_prev, C)

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


def AuxiliaryHeadCIFAR(input, num_classes, aux_name='auxiliary_head'):
    relu_a = fluid.layers.relu(input)
    pool_a = fluid.layers.pool2d(relu_a, 5, 'avg', 3)
    conv2d_a = fluid.layers.conv2d(
        pool_a,
        128,
        1,
        name=aux_name + '.features.2',
        param_attr=ParamAttr(
            initializer=Xavier(
                uniform=False, fan_in=0),
            name=aux_name + '.features.2.weight'),
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
        param_attr=ParamAttr(
            initializer=Xavier(
                uniform=False, fan_in=0),
            name=aux_name + '.features.5.weight'),
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
    fc_name = aux_name + '.classifier'
    fc = fluid.layers.fc(bn_b,
                         num_classes,
                         name=fc_name,
                         param_attr=ParamAttr(
                             initializer=Normal(scale=1e-3),
                             name=fc_name + '.weight'),
                         bias_attr=ParamAttr(
                             initializer=Constant(0.), name=fc_name + '.bias'))
    return fc


def StemConv(input, C_out, kernel_size, padding):
    conv_a = fluid.layers.conv2d(
        input,
        C_out,
        kernel_size,
        padding=padding,
        param_attr=ParamAttr(
            initializer=Xavier(
                uniform=False, fan_in=0), name='stem.0.weight'),
        bias_attr=False)
    bn_a = fluid.layers.batch_norm(
        conv_a,
        param_attr=ParamAttr(
            initializer=Constant(1.), name='stem.1.weight'),
        bias_attr=ParamAttr(
            initializer=Constant(0.), name='stem.1.bias'),
        moving_mean_name='stem.1.running_mean',
        moving_variance_name='stem.1.running_var')
    return bn_a


class NetworkCIFAR(object):
    def __init__(self, C, class_num, layers, auxiliary, genotype):
        self.class_num = class_num
        self._layers = layers
        self._auxiliary = auxiliary

        stem_multiplier = 3
        self.drop_path_prob = 0
        C_curr = stem_multiplier * C

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = []
        reduction_prev = False
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

    def forward(self, init_channel, is_train):
        self.training = is_train
        self.logits_aux = None
        num_channel = init_channel * 3
        s0 = StemConv(self.image, num_channel, kernel_size=3, padding=1)
        s1 = s0
        for i, cell in enumerate(self.cells):
            name = 'cells.' + str(i) + '.'
            s0, s1 = s1, cell.forward(s0, s1, self.drop_path_prob, is_train,
                                      name)
            if i == int(2 * self._layers // 3):
                if self._auxiliary and self.training:
                    self.logits_aux = AuxiliaryHeadCIFAR(s1, self.class_num)
        out = fluid.layers.adaptive_pool2d(s1, (1, 1), "avg")
        self.logits = fluid.layers.fc(out,
                                      size=self.class_num,
                                      param_attr=ParamAttr(
                                          initializer=Normal(scale=1e-3),
                                          name='classifier.weight'),
                                      bias_attr=ParamAttr(
                                          initializer=Constant(0.),
                                          name='classifier.bias'))
        return self.logits, self.logits_aux

    def build_input(self, image_shape, batch_size, is_train):
        if is_train:
            py_reader = fluid.layers.py_reader(
                capacity=64,
                shapes=[[-1] + image_shape, [-1, 1], [-1, 1], [-1, 1], [-1, 1],
                        [-1, 1], [-1, batch_size, self.class_num - 1]],
                lod_levels=[0, 0, 0, 0, 0, 0, 0],
                dtypes=[
                    "float32", "int64", "int64", "float32", "int32", "int32",
                    "float32"
                ],
                use_double_buffer=True,
                name='train_reader')
        else:
            py_reader = fluid.layers.py_reader(
                capacity=64,
                shapes=[[-1] + image_shape, [-1, 1]],
                lod_levels=[0, 0],
                dtypes=["float32", "int64"],
                use_double_buffer=True,
                name='test_reader')
        return py_reader

    def train_model(self, py_reader, init_channels, aux, aux_w, batch_size,
                    loss_lambda):
        self.image, self.ya, self.yb, self.lam, self.label_reshape,\
           self.non_label_reshape, self.rad_var = fluid.layers.read_file(py_reader)
        self.logits, self.logits_aux = self.forward(init_channels, True)
        self.mixup_loss = self.mixup_loss(aux, aux_w)
        self.lrc_loss = self.lrc_loss(batch_size)
        return self.mixup_loss + loss_lambda * self.lrc_loss

    def test_model(self, py_reader, init_channels):
        self.image, self.ya = fluid.layers.read_file(py_reader)
        self.logits, _ = self.forward(init_channels, False)
        prob = fluid.layers.softmax(self.logits, use_cudnn=False)
        loss = fluid.layers.cross_entropy(prob, self.ya)
        acc_1 = fluid.layers.accuracy(self.logits, self.ya, k=1)
        acc_5 = fluid.layers.accuracy(self.logits, self.ya, k=5)
        return loss, acc_1, acc_5

    def mixup_loss(self, auxiliary, auxiliary_weight):
        prob = fluid.layers.softmax(self.logits, use_cudnn=False)
        loss_a = fluid.layers.cross_entropy(prob, self.ya)
        loss_b = fluid.layers.cross_entropy(prob, self.yb)
        loss_a_mean = fluid.layers.reduce_mean(loss_a)
        loss_b_mean = fluid.layers.reduce_mean(loss_b)
        loss = self.lam * loss_a_mean + (1 - self.lam) * loss_b_mean
        if auxiliary:
            prob_aux = fluid.layers.softmax(self.logits_aux, use_cudnn=False)
            loss_a_aux = fluid.layers.cross_entropy(prob_aux, self.ya)
            loss_b_aux = fluid.layers.cross_entropy(prob_aux, self.yb)
            loss_a_aux_mean = fluid.layers.reduce_mean(loss_a_aux)
            loss_b_aux_mean = fluid.layers.reduce_mean(loss_b_aux)
            loss_aux = self.lam * loss_a_aux_mean + (1 - self.lam
                                                     ) * loss_b_aux_mean
        return loss + auxiliary_weight * loss_aux

    def lrc_loss(self, batch_size):
        y_diff_reshape = fluid.layers.reshape(self.logits, shape=(-1, 1))
        label_reshape = fluid.layers.squeeze(self.label_reshape, axes=[1])
        non_label_reshape = fluid.layers.squeeze(
            self.non_label_reshape, axes=[1])
        label_reshape.stop_gradient = True
        non_label_reshape.stop_graident = True

        y_diff_label_reshape = fluid.layers.gather(y_diff_reshape,
                                                   label_reshape)
        y_diff_non_label_reshape = fluid.layers.gather(y_diff_reshape,
                                                       non_label_reshape)
        y_diff_label = fluid.layers.reshape(
            y_diff_label_reshape, shape=(-1, batch_size, 1))
        y_diff_non_label = fluid.layers.reshape(
            y_diff_non_label_reshape,
            shape=(-1, batch_size, self.class_num - 1))
        y_diff_ = y_diff_non_label - y_diff_label

        y_diff_ = fluid.layers.transpose(y_diff_, perm=[1, 2, 0])
        rad_var_trans = fluid.layers.transpose(self.rad_var, perm=[1, 2, 0])
        rad_y_diff_trans = rad_var_trans * y_diff_
        lrc_loss_sum = fluid.layers.reduce_sum(rad_y_diff_trans, dim=[0, 1])
        lrc_loss_ = fluid.layers.abs(lrc_loss_sum) / (batch_size *
                                                      (self.class_num - 1))
        lrc_loss_mean = fluid.layers.reduce_mean(lrc_loss_)

        return lrc_loss_mean
