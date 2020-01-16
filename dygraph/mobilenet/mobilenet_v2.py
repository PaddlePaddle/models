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

import os
import numpy as np
import time
import sys
import sys
import numpy as np
import argparse
import ast
import paddle
import paddle.fluid as fluid
from paddle.fluid.initializer import MSRA
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, BatchNorm, FC
from paddle.fluid.dygraph.base import to_variable

from paddle.fluid import framework

import math
import sys


class ConvBNLayer(fluid.dygraph.Layer):
    def __init__(self,
                 filter_size,
                 num_filters,
                 stride,
                 padding,
                 channels=None,
                 num_groups=1,
                 name=None,
                 use_cudnn=True):
        super(ConvBNLayer, self).__init__(name)

        tmp_param = ParamAttr(name=name + "_weights")
        self._conv = Conv2D(
            self.full_name(),
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
            self.full_name(),
            num_filters,
            param_attr=ParamAttr(name=name + "_bn" + "_scale"),
            bias_attr=ParamAttr(name=name + "_bn" + "_offset"),
            moving_mean_name=name + "_bn" + '_mean',
            moving_variance_name=name + "_bn" + '_variance')

    def forward(self, inputs, if_act=True):
        y = self._conv(inputs)
        y = self._batch_norm(y)
        if if_act:
            y = fluid.layers.relu6(y)
        return y


class InvertedResidualUnit(fluid.dygraph.Layer):
    def __init__(self,
                 num_in_filter,
                 num_filters,
                 stride,
                 filter_size,
                 padding,
                 expansion_factor,
                 name=None):
        super(InvertedResidualUnit, self).__init__(name)
        num_expfilter = int(round(num_in_filter * expansion_factor))
        self._expand_conv = ConvBNLayer(
            name=name + "_expand",
            num_filters=num_expfilter,
            filter_size=1,
            stride=1,
            padding=0,
            num_groups=1)

        self._bottleneck_conv = ConvBNLayer(
            name=name + "_dwise",
            num_filters=num_expfilter,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            num_groups=num_expfilter,
            use_cudnn=False)

        self._linear_conv = ConvBNLayer(
            name=name + "_linear",
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
    def __init__(self, in_c, t, c, n, s, name=None):
        super(InvresiBlocks, self).__init__(name)

        self._first_block = InvertedResidualUnit(
            name=name + "_1",
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
                    name=name + "_" + str(i + 1),
                    num_in_filter=c,
                    num_filters=c,
                    stride=1,
                    filter_size=3,
                    padding=1,
                    expansion_factor=t),
                name=name + "_" + str(i + 1))
            self._inv_blocks.append(tmp)

    def forward(self, inputs):
        y = self._first_block(inputs, ifshortcut=False)
        for inv_block in self._inv_blocks:
            y = inv_block(y, ifshortcut=True)
        return y


class MobileNetV2(fluid.dygraph.Layer):
    def __init__(self, name, class_dim=1000, scale=1.0):
        super(MobileNetV2, self).__init__(name)
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
            name="conv1_1",
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
                    name='conv' + str(i),
                    in_c=in_c,
                    t=t,
                    c=int(c * scale),
                    n=n,
                    s=s),
                name='conv' + str(i))
            self._invl.append(tmp)
            in_c = int(c * scale)

        #3. last_conv
        self._conv9 = ConvBNLayer(
            name="conv9",
            num_filters=int(1280 * scale) if scale > 1.0 else 1280,
            filter_size=1,
            stride=1,
            padding=0)

        #4. pool
        self._pool2d_avg = Pool2D(
            name_scope="pool", pool_type='avg', global_pooling=True)

        #5. fc
        tmp_param = ParamAttr(name="fc10_weights")
        self._fc = FC(name_scope="fc",
                      size=class_dim,
                      param_attr=tmp_param,
                      bias_attr=ParamAttr(name="fc10_offset"))

    def forward(self, inputs):
        y = self._conv1(inputs, if_act=True)
        for inv in self._invl:
            y = inv(y)
        y = self._conv9(y, if_act=True)
        y = self._pool2d_avg(y)
        y = self._fc(y)
        return y
