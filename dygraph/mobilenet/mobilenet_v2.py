# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#order: standard library, third party, local library 
import os
import time
import math
import sys
import numpy as np
import argparse
import paddle
import paddle.fluid as fluid
from paddle.fluid.initializer import MSRA
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, BatchNorm, Linear
from paddle.fluid.dygraph.base import to_variable
from paddle.fluid import framework



class ConvBNLayer(fluid.dygraph.Layer):
    def __init__(self,
                 num_channels,
                 filter_size,
                 num_filters,
                 stride,
                 padding,
                 channels=None,
                 num_groups=1,
                 use_cudnn=True):
        super(ConvBNLayer, self).__init__()

        tmp_param = ParamAttr(name=self.full_name() + "_weights")
        self._conv = Conv2D(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            groups=num_groups,
            act=None,
            use_cudnn=use_cudnn,
            param_attr=tmp_param,
            bias_attr=False)

        self._batch_norm = BatchNorm(
            num_filters,
            param_attr=ParamAttr(name=self.full_name() + "_bn" + "_scale"),
            bias_attr=ParamAttr(name=self.full_name() + "_bn" + "_offset"),
            moving_mean_name=self.full_name() + "_bn" + '_mean',
            moving_variance_name=self.full_name() + "_bn" + '_variance')

    def forward(self, inputs, if_act=True):
        y = self._conv(inputs)
        y = self._batch_norm(y)
        if if_act:
            y = fluid.layers.relu6(y)
        return y


class InvertedResidualUnit(fluid.dygraph.Layer):
    def __init__(
            self,
            num_channels,
            num_in_filter,
            num_filters,
            stride,
            filter_size,
            padding,
            expansion_factor, ):
        super(InvertedResidualUnit, self).__init__()
        num_expfilter = int(round(num_in_filter * expansion_factor))
        self._expand_conv = ConvBNLayer(
            num_channels=num_channels,
            num_filters=num_expfilter,
            filter_size=1,
            stride=1,
            padding=0,
            num_groups=1)

        self._bottleneck_conv = ConvBNLayer(
            num_channels=num_expfilter,
            num_filters=num_expfilter,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            num_groups=num_expfilter,
            use_cudnn=False)

        self._linear_conv = ConvBNLayer(
            num_channels=num_expfilter,
            num_filters=num_filters,
            filter_size=1,
            stride=1,
            padding=0,
            num_groups=1)

    def forward(self, inputs, ifshortcut):
        y = self._expand_conv(inputs, if_act=True)
        y = self._bottleneck_conv(y, if_act=True)
        y = self._linear_conv(y, if_act=False)
        if ifshortcut:
            y = fluid.layers.elementwise_add(inputs, y)
        return y


class InvresiBlocks(fluid.dygraph.Layer):
    def __init__(self, in_c, t, c, n, s):
        super(InvresiBlocks, self).__init__()

        self._first_block = InvertedResidualUnit(
            num_channels=in_c,
            num_in_filter=in_c,
            num_filters=c,
            stride=s,
            filter_size=3,
            padding=1,
            expansion_factor=t)

        self._inv_blocks = []
        for i in range(1, n):
            tmp = self.add_sublayer(
                sublayer=InvertedResidualUnit(
                    num_channels=c,
                    num_in_filter=c,
                    num_filters=c,
                    stride=1,
                    filter_size=3,
                    padding=1,
                    expansion_factor=t),
                name=self.full_name() + "_" + str(i + 1))
            self._inv_blocks.append(tmp)

    def forward(self, inputs):
        y = self._first_block(inputs, ifshortcut=False)
        for inv_block in self._inv_blocks:
            y = inv_block(y, ifshortcut=True)
        return y


class MobileNetV2(fluid.dygraph.Layer):
    def __init__(self, class_dim=1000, scale=1.0):
        super(MobileNetV2, self).__init__()
        self.scale = scale
        self.class_dim = class_dim

        bottleneck_params_list = [
            (1, 16, 1, 1),
            (6, 24, 2, 2),
            (6, 32, 3, 2),
            (6, 64, 4, 2),
            (6, 96, 3, 1),
            (6, 160, 3, 2),
            (6, 320, 1, 1),
        ]

        #1. conv1 
        self._conv1 = ConvBNLayer(
            num_channels=3,
            num_filters=int(32 * scale),
            filter_size=3,
            stride=2,
            padding=1)

        #2. bottleneck sequences
        self._invl = []
        i = 1
        in_c = int(32 * scale)
        for layer_setting in bottleneck_params_list:
            t, c, n, s = layer_setting
            i += 1
            tmp = self.add_sublayer(
                sublayer=InvresiBlocks(
                    in_c=in_c, t=t, c=int(c * scale), n=n, s=s),
                name='conv' + str(i))
            self._invl.append(tmp)
            in_c = int(c * scale)

        #3. last_conv
        self._out_c = int(1280 * scale) if scale > 1.0 else 1280
        self._conv9 = ConvBNLayer(
            num_channels=in_c,
            num_filters=self._out_c,
            filter_size=1,
            stride=1,
            padding=0)

        #4. pool
        self._pool2d_avg = Pool2D(pool_type='avg', global_pooling=True)

        #5. fc
        tmp_param = ParamAttr(name=self.full_name() + "fc10_weights")
        self._fc = Linear(
            self._out_c,
            class_dim,
            param_attr=tmp_param,
            bias_attr=ParamAttr(name="fc10_offset"))

    def forward(self, inputs):
        y = self._conv1(inputs, if_act=True)
        for inv in self._invl:
            y = inv(y)
        y = self._conv9(y, if_act=True)
        y = self._pool2d_avg(y)
        y = fluid.layers.reshape(y, shape=[-1, self._out_c])
        y = self._fc(y)
        return y
