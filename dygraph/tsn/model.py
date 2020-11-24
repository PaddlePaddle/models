# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import paddle
import math
from paddle.nn import Conv2D, BatchNorm2D, Linear, Dropout, MaxPool2D, AvgPool2D
from paddle import ParamAttr
import paddle.nn.functional as F
from paddle.jit import to_static
from paddle.static import InputSpec


class ConvBNLayer(paddle.nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 groups=1,
                 act=None,
                 name=None):
        super(ConvBNLayer, self).__init__()
        self._conv = Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            groups=groups,
            weight_attr=ParamAttr(name=name + "_weights"),
            bias_attr=False)
        if name == "conv1":
            bn_name = "bn_" + name
        else:
            bn_name = "bn" + name[3:]

        self._act = act

        self._batch_norm = BatchNorm2D(
            out_channels,
            weight_attr=ParamAttr(name=bn_name + "_scale"),
            bias_attr=ParamAttr(bn_name + "_offset"))

    def forward(self, inputs):
        y = self._conv(inputs)

        y = self._batch_norm(y)
        if self._act:
            y = getattr(paddle.nn.functional, self._act)(y)
        return y


class BottleneckBlock(paddle.nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 shortcut=True,
                 name=None):
        super(BottleneckBlock, self).__init__()
        self.conv0 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            act="relu",
            name=name + "_branch2a")
        self.conv1 = ConvBNLayer(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            act="relu",
            name=name + "_branch2b")

        self.conv2 = ConvBNLayer(
            in_channels=out_channels,
            out_channels=out_channels * 4,
            kernel_size=1,
            act=None,
            name=name + "_branch2c")

        if not shortcut:
            self.short = ConvBNLayer(
                in_channels=in_channels,
                out_channels=out_channels * 4,
                kernel_size=1,
                stride=stride,
                name=name + "_branch1")

        self.shortcut = shortcut

    def forward(self, inputs):
        y = self.conv0(inputs)
        conv1 = self.conv1(y)
        conv2 = self.conv2(conv1)

        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)

        y = paddle.add(x=short, y=conv2)
        return F.relu(y)


class BasicBlock(paddle.nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 shortcut=True,
                 name=None):
        super(BasicBlock, self).__init__()
        self.stride = stride
        self.conv0 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            filter_size=3,
            stride=stride,
            act="relu",
            name=name + "_branch2a")
        self.conv1 = ConvBNLayer(
            in_channels=out_channels,
            out_channels=out_channels,
            filter_size=3,
            act=None,
            name=name + "_branch2b")

        if not shortcut:
            self.short = ConvBNLayer(
                in_channels=in_channels,
                out_channels=out_channels,
                filter_size=1,
                stride=stride,
                name=name + "_branch1")

        self.shortcut = shortcut

    def forward(self, inputs):
        y = self.conv0(inputs)
        conv1 = self.conv1(y)

        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)
        y = paddle.add(short, conv1)
        y = F.relu(y)
        return y


class TSN_ResNet(paddle.nn.Layer):
    def __init__(self, config):
        super(TSN_ResNet, self).__init__()
        self.layers = config.MODEL.num_layers
        self.seg_num = config.MODEL.seg_num
        self.class_dim = config.MODEL.num_classes
        supported_layers = [18, 34, 50, 101, 152]
        assert self.layers in supported_layers, \
            "supported layers are {} but input layer is {}".format(
                supported_layers, layers)

        if self.layers == 18:
            depth = [2, 2, 2, 2]
        elif self.layers == 34 or self.layers == 50:
            depth = [3, 4, 6, 3]
        elif self.layers == 101:
            depth = [3, 4, 23, 3]
        elif self.layers == 152:
            depth = [3, 8, 36, 3]
        in_channels = [64, 256, 512,
                       1024] if self.layers >= 50 else [64, 64, 128, 256]
        out_channels = [64, 128, 256, 512]

        self.conv = ConvBNLayer(
            in_channels=3,
            out_channels=64,
            kernel_size=7,
            stride=2,
            act="relu",
            name="conv1")
        self.pool2D_max = MaxPool2D(kernel_size=3, stride=2, padding=1)

        self.block_list = []
        if self.layers >= 50:
            for block in range(len(depth)):
                shortcut = False
                for i in range(depth[block]):
                    if self.layers in [101, 152] and block == 2:
                        if i == 0:
                            conv_name = "res" + str(block + 2) + "a"
                        else:
                            conv_name = "res" + str(block + 2) + "b" + str(i)
                    else:
                        conv_name = "res" + str(block + 2) + chr(97 + i)
                    bottleneck_block = self.add_sublayer(
                        conv_name,
                        BottleneckBlock(
                            in_channels=in_channels[block]
                            if i == 0 else out_channels[block] * 4,
                            out_channels=out_channels[block],
                            stride=2 if i == 0 and block != 0 else 1,
                            shortcut=shortcut,
                            name=conv_name))
                    self.block_list.append(bottleneck_block)
                    shortcut = True
        else:
            for block in range(len(depth)):
                shortcut = False
                for i in range(depth[block]):
                    conv_name = "res" + str(block + 2) + chr(97 + i)
                    basic_block = self.add_sublayer(
                        conv_name,
                        BasicBlock(
                            in_channels=in_channels[block]
                            if i == 0 else out_channels[block],
                            out_channels=out_channels[block],
                            stride=2 if i == 0 and block != 0 else 1,
                            shortcut=shortcut,
                            name=conv_name))
                    self.block_list.append(basic_block)
                    shortcut = True

        self.pool2D_avg = AvgPool2D(kernel_size=7)

        self.pool2D_avg_channels = in_channels[-1] * 2

        self.out = Linear(
            self.pool2D_avg_channels,
            self.class_dim,
            weight_attr=ParamAttr(
                initializer=paddle.nn.initializer.Normal(
                    mean=0.0, std=0.01),
                name="fc_0.w_0"),
            bias_attr=ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=0.0),
                name="fc_0.b_0"))

    #@to_static(input_spec=[InputSpec(shape=[None, 3, 224, 224], name='inputs')])

    def forward(self, inputs):
        y = paddle.reshape(
            inputs, [-1, inputs.shape[2], inputs.shape[3], inputs.shape[4]])
        y = self.conv(y)
        y = self.pool2D_max(y)
        for block in self.block_list:
            y = block(y)
        y = self.pool2D_avg(y)
        y = F.dropout(y, p=0.2)
        y = paddle.reshape(y, [-1, self.seg_num, y.shape[1]])
        y = paddle.mean(y, axis=1)
        y = paddle.reshape(y, shape=[-1, 2048])
        y = self.out(y)
        y = F.softmax(y)
        return y
