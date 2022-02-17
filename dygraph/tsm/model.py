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

import os
import time
import sys
import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, BatchNorm, Linear
import math


class ConvBNLayer(fluid.dygraph.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 filter_size,
                 stride=1,
                 groups=1,
                 act=None,
                 name=None):
        super(ConvBNLayer, self).__init__()

        self._conv = Conv2D(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=None,
            act=None,
            param_attr=fluid.param_attr.ParamAttr(name=name + "_weights"),
            bias_attr=False)
        if name == "conv1":
            bn_name = "bn_" + name
        else:
            bn_name = "bn" + name[3:]

        self._batch_norm = BatchNorm(
            num_filters,
            act=act,
            param_attr=ParamAttr(
                name=bn_name + "_scale"),  #fluid.param_attr.ParamAttr(),
            bias_attr=ParamAttr(bn_name +
                                "_offset"),  #fluid.param_attr.ParamAttr())
            moving_mean_name=bn_name + "_mean",
            moving_variance_name=bn_name + "_variance")

    def forward(self, inputs):
        y = self._conv(inputs)
        y = self._batch_norm(y)

        return y


class BottleneckBlock(fluid.dygraph.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 stride,
                 shortcut=True,
                 seg_num=8,
                 name=None):
        super(BottleneckBlock, self).__init__()
        self.conv0 = ConvBNLayer(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=1,
            act='relu',
            name=name + "_branch2a")
        self.conv1 = ConvBNLayer(
            num_channels=num_filters,
            num_filters=num_filters,
            filter_size=3,
            stride=stride,
            act='relu',
            name=name + "_branch2b")
        self.conv2 = ConvBNLayer(
            num_channels=num_filters,
            num_filters=num_filters * 4,
            filter_size=1,
            act=None,
            name=name + "_branch2c")

        if not shortcut:
            self.short = ConvBNLayer(
                num_channels=num_channels,
                num_filters=num_filters * 4,
                filter_size=1,
                stride=stride,
                name=name + "_branch1")
        self.shortcut = shortcut
        self.seg_num = seg_num
        self._num_channels_out = int(num_filters * 4)

    def forward(self, inputs):
        shifts = fluid.layers.temporal_shift(inputs, self.seg_num, 1.0 / 8)
        y = self.conv0(shifts)
        conv1 = self.conv1(y)
        conv2 = self.conv2(conv1)
        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)
        y = fluid.layers.elementwise_add(x=short, y=conv2, act="relu")
        return y


class TSM_ResNet(fluid.dygraph.Layer):
    def __init__(self, name_scope, config):
        super(TSM_ResNet, self).__init__(name_scope)

        self.layers = config.MODEL.num_layers
        self.seg_num = config.MODEL.seg_num
        self.class_dim = config.MODEL.num_classes

        if self.layers == 50:
            depth = [3, 4, 6, 3]
        else:
            raise NotImplementedError
        num_filters = [64, 128, 256, 512]

        self.conv = ConvBNLayer(
            num_channels=3,
            num_filters=64,
            filter_size=7,
            stride=2,
            act='relu',
            name="conv1")
        self.pool2d_max = Pool2D(
            pool_size=3, pool_stride=2, pool_padding=1, pool_type='max')

        self.bottleneck_block_list = []
        num_channels = 64

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
                        num_channels=num_channels,
                        num_filters=num_filters[block],
                        stride=2 if i == 0 and block != 0 else 1,
                        shortcut=shortcut,
                        seg_num=self.seg_num,
                        name=conv_name))
                num_channels = int(bottleneck_block._num_channels_out)
                self.bottleneck_block_list.append(bottleneck_block)
                shortcut = True
        self.pool2d_avg = Pool2D(
            pool_size=7, pool_type='avg', global_pooling=True)

        import math
        stdv = 1.0 / math.sqrt(2048 * 1.0)

        self.out = Linear(
            2048,
            self.class_dim,
            act="softmax",
            param_attr=fluid.param_attr.ParamAttr(
                initializer=fluid.initializer.Uniform(-stdv, stdv),
                name="fc_0.w_0"),
            bias_attr=fluid.param_attr.ParamAttr(
                learning_rate=2.0,
                regularizer=fluid.regularizer.L2Decay(0.),
                name="fc_0.b_0"))

    def forward(self, inputs):
        y = fluid.layers.reshape(
            inputs, [-1, inputs.shape[2], inputs.shape[3], inputs.shape[4]])
        y = self.conv(y)
        y = self.pool2d_max(y)
        for bottleneck_block in self.bottleneck_block_list:
            y = bottleneck_block(y)
        y = self.pool2d_avg(y)
        y = fluid.layers.dropout(y, dropout_prob=0.5)
        y = fluid.layers.reshape(y, [-1, self.seg_num, y.shape[1]])
        y = fluid.layers.reduce_mean(y, dim=1)
        y = fluid.layers.reshape(y, shape=[-1, 2048])
        y = self.out(y)
        return y
