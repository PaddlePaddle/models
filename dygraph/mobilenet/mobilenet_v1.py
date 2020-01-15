# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
import time
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
                 name_scope,
                 filter_size,
                 num_filters,
                 stride,
                 padding,
                 channels=None,
                 num_groups=1,
                 act='relu',
                 use_cudnn=True,
                 name=None):
        super(ConvBNLayer, self).__init__(name_scope)

        self._conv = Conv2D(
            self.full_name(),
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            groups=num_groups,
            act=None,
            use_cudnn=use_cudnn,
            param_attr=ParamAttr(
                initializer=MSRA(), name=self.full_name() + "_weights"),
            bias_attr=False)

        self._batch_norm = BatchNorm(
            self.full_name(),
            num_filters,
            act=act,
            param_attr=ParamAttr(name="_bn" + "_scale"),
            bias_attr=ParamAttr(name="_bn" + "_offset"),
            moving_mean_name="_bn" + '_mean',
            moving_variance_name="_bn" + '_variance')

    def forward(self, inputs):
        y = self._conv(inputs)
        y = self._batch_norm(y)
        return y


class DepthwiseSeparable(fluid.dygraph.Layer):
    def __init__(self,
                 name_scope,
                 num_filters1,
                 num_filters2,
                 num_groups,
                 stride,
                 scale,
                 name=None):
        super(DepthwiseSeparable, self).__init__(name_scope)

        self._depthwise_conv = ConvBNLayer(
            name_scope="dw",
            num_filters=int(num_filters1 * scale),
            filter_size=3,
            stride=stride,
            padding=1,
            num_groups=int(num_groups * scale),
            use_cudnn=False)

        self._pointwise_conv = ConvBNLayer(
            name_scope="sep",
            filter_size=1,
            num_filters=int(num_filters2 * scale),
            stride=1,
            padding=0)

    def forward(self, inputs):
        y = self._depthwise_conv(inputs)
        y = self._pointwise_conv(y)
        return y


class MobileNetV1(fluid.dygraph.Layer):
    def __init__(self, name_scope, scale=1.0, class_dim=102):
        super(MobileNetV1, self).__init__(name_scope)
        self.scale = scale
        self.dwsl = []

        self.conv1 = ConvBNLayer(
            name_scope="conv1",
            filter_size=3,
            channels=3,
            num_filters=int(32 * scale),
            stride=2,
            padding=1)

        dws21 = self.add_sublayer(
            sublayer=DepthwiseSeparable(
                name_scope="conv2_1",
                num_filters1=32,
                num_filters2=64,
                num_groups=32,
                stride=1,
                scale=scale),
            name="conv2_1")
        self.dwsl.append(dws21)

        dws22 = self.add_sublayer(
            sublayer=DepthwiseSeparable(
                name_scope="conv2_2",
                num_filters1=64,
                num_filters2=128,
                num_groups=64,
                stride=2,
                scale=scale),
            name="conv2_2")
        self.dwsl.append(dws22)

        dws31 = self.add_sublayer(
            sublayer=DepthwiseSeparable(
                name_scope="conv3_1",
                num_filters1=128,
                num_filters2=128,
                num_groups=128,
                stride=1,
                scale=scale),
            name="conv3_1")
        self.dwsl.append(dws31)

        dws32 = self.add_sublayer(
            sublayer=DepthwiseSeparable(
                name_scope="conv3_2",
                num_filters1=128,
                num_filters2=256,
                num_groups=128,
                stride=2,
                scale=scale),
            name="conv3_2")
        self.dwsl.append(dws32)

        dws41 = self.add_sublayer(
            sublayer=DepthwiseSeparable(
                name_scope="conv4_1",
                num_filters1=256,
                num_filters2=256,
                num_groups=256,
                stride=1,
                scale=scale),
            name="conv4_1")
        self.dwsl.append(dws41)

        dws42 = self.add_sublayer(
            sublayer=DepthwiseSeparable(
                name_scope="conv4_2",
                num_filters1=256,
                num_filters2=512,
                num_groups=256,
                stride=2,
                scale=scale),
            name="conv4_2")
        self.dwsl.append(dws42)

        for i in range(5):
            tmp = self.add_sublayer(
                sublayer=DepthwiseSeparable(
                    name_scope="conv5_" + str(i + 1),
                    num_filters1=512,
                    num_filters2=512,
                    num_groups=512,
                    stride=1,
                    scale=scale),
                name="conv5_" + str(i + 1))
            self.dwsl.append(tmp)

        dws56 = self.add_sublayer(
            sublayer=DepthwiseSeparable(
                name_scope="conv5_6",
                num_filters1=512,
                num_filters2=1024,
                num_groups=512,
                stride=2,
                scale=scale),
            name="conv5_6")
        self.dwsl.append(dws56)

        dws6 = self.add_sublayer(
            sublayer=DepthwiseSeparable(
                name_scope="conv6",
                num_filters1=1024,
                num_filters2=1024,
                num_groups=1024,
                stride=1,
                scale=scale),
            name="conv6")
        self.dwsl.append(dws6)

        self.pool2d_avg = Pool2D(
            name_scope="pool", pool_type='avg', global_pooling=True)

        self.out = FC(name_scope="fc",
                      size=class_dim,
                      param_attr=ParamAttr(
                          initializer=MSRA(),
                          name=self.full_name() + "fc7_weights"),
                      bias_attr=ParamAttr(name="fc7_offset"))

    def forward(self, inputs):
        y = self.conv1(inputs)
        idx = 0
        for dws in self.dwsl:
            y = dws(y)
        y = self.pool2d_avg(y)
        y = self.out(y)
        return y
