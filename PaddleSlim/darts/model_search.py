# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import numpy as np
import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.initializer import NormalInitializer, UniformInitializer, ConstantInitializer
from genotypes import PRIMITIVES
from genotypes import Genotype
from operations import *


def mixed_op(x, c_out, stride, index, reduction, name):
    param_attr = ParamAttr(
        name="arch/weight{}_{}".format(2 if reduction else 1, index))
    weight = fluid.layers.create_parameter(
        shape=[len(PRIMITIVES)],
        dtype="float32",
        attr=param_attr,
        default_initializer=NormalInitializer(
            loc=0.0, scale=1e-3))
    weight = fluid.layers.softmax(weight)
    ops = []
    index = 0
    for primitive in PRIMITIVES:
        op = OPS[primitive](x, c_out, stride, False, name)
        if 'pool' in primitive:
            gama = ParamAttr(
                name=name + '/' + primitive + "/mixed_bn_gama",
                initializer=fluid.initializer.Constant(value=1),
                trainable=False)
            beta = ParamAttr(
                name=name + '/' + primitive + "/mixed_bn_beta",
                initializer=fluid.initializer.Constant(value=0),
                trainable=False)
            op = fluid.layers.batch_norm(
                op,
                param_attr=gama,
                bias_attr=beta,
                moving_mean_name=name + '/' + primitive + "/mixed_bn_mean",
                moving_variance_name=name + '/' + primitive +
                "/mixed_bn_variance")
        ops.append(fluid.layers.elementwise_mul(op, weight[index]))
        index += 1
    out = fluid.layers.sums(ops)
    return out


def cell(s0, s1, steps, multiplier, c_out, reduction, reduction_prev, name):
    if reduction_prev:
        s0 = factorized_reduce(s0, c_out, False, name + "/s-2")
    else:
        s0 = relu_conv_bn(s0, c_out, 1, 1, 0, False, name + "/s-2")
    s1 = relu_conv_bn(s1, c_out, 1, 1, 0, False, name + '/s-1')
    state = [s0, s1]
    offset = 0
    for i in range(steps):
        temp = []
        for j in range(2 + i):
            stride = 2 if reduction and j < 2 else 1
            temp.append(
                mixed_op(state[j], c_out, stride, offset + j, reduction, name +
                         "/s" + str(offset + j)))
        offset += len(state)
        state.append(fluid.layers.sums(temp))
    out = fluid.layers.concat(input=state[-multiplier:], axis=1)
    return out


def model(x,
          y,
          c_in,
          num_classes,
          layers,
          steps=4,
          multiplier=4,
          stem_multiplier=3,
          name="model"):
    c_curr = stem_multiplier * c_in
    k = (1. / x.shape[1] / 3 / 3)**0.5
    x = fluid.layers.conv2d(
        x,
        c_curr,
        3,
        padding=1,
        param_attr=fluid.ParamAttr(
            name=name + "/conv_0",
            initializer=UniformInitializer(
                low=-k, high=k)),
        bias_attr=False)
    x = fluid.layers.batch_norm(
        x,
        param_attr=fluid.ParamAttr(
            name=name + "/bn0_scale", initializer=ConstantInitializer(value=1)),
        bias_attr=fluid.ParamAttr(
            name=name + "/bn0_offset",
            initializer=ConstantInitializer(value=0)),
        moving_mean_name=name + "/bn0_mean",
        moving_variance_name=name + "/bn0_variance")
    s0 = s1 = x
    reduction_prev = False
    c_curr = c_in
    for i in range(layers):
        if i in [layers // 3, 2 * layers // 3]:
            c_curr *= 2
            reduction = True
        else:
            reduction = False
        s0, s1 = s1, cell(s0, s1, steps, multiplier, c_curr, reduction,
                          reduction_prev, name + "/l" + str(i))
        reduction_prev = reduction
    out = fluid.layers.pool2d(s1, pool_type='avg', global_pooling=True)
    out = fluid.layers.squeeze(out, axes=[2, 3])
    k = (1. / out.shape[1])**0.5
    logits = fluid.layers.fc(out,
                             num_classes,
                             param_attr=fluid.ParamAttr(
                                 name=name + "/fc_weights",
                                 initializer=UniformInitializer(
                                     low=-k, high=k)),
                             bias_attr=fluid.ParamAttr(
                                 name=name + "/fc_bias",
                                 initializer=UniformInitializer(
                                     low=-k, high=k)))
    train_loss = fluid.layers.reduce_mean(
        fluid.layers.softmax_with_cross_entropy(logits, y))
    return logits, train_loss


def get_genotype(arch_names, arch_values, steps=4, multiplier=4):
    def _parse(stride):
        genotype = []
        offset = 0
        for i in range(steps):
            edges = []
            edges_confident = []
            for j in range(i + 2):
                value = arch_values[arch_names.index("arch/weight{}_{}".format(
                    stride, offset + j))]
                value_sorted = value.argsort()
                max_index = value_sorted[-2] if value_sorted[
                    -1] == PRIMITIVES.index('none') else value_sorted[-1]

                edges.append((PRIMITIVES[max_index], j))
                edges_confident.append(value[max_index])

            edges_confident = np.array(edges_confident)
            max_edges = [
                edges[np.argsort(edges_confident)[-1]],
                edges[np.argsort(edges_confident)[-2]]
            ]
            genotype.extend(max_edges)
            offset += i + 2
        return genotype

    concat = list(range(2 + steps - multiplier, steps + 2))
    gene_normal = _parse(1)
    gene_reduce = _parse(2)
    genotype = Genotype(
        normal=gene_normal,
        normal_concat=concat,
        reduce=gene_reduce,
        reduce_concat=concat)
    return genotype
