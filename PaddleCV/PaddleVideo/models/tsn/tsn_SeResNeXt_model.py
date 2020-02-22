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

import os
import time
import sys
import paddle.fluid as fluid
import math


class TSN_SeResNeXt():
    def __init__(self, layers=152, seg_num=7, is_training=True):
        self.layers = layers
        self.seg_num = seg_num
        self.is_training = is_training

    def conv_bn_layer(self,
                      input,
                      num_filters,
                      filter_size,
                      stride=1,
                      groups=1,
                      act=None,
                      name=None):
        n = filter_size * filter_size * num_filters
        std = math.sqrt(2. / n)
        conv = fluid.layers.conv2d(
            input=input,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=groups,
            act=None,
            param_attr=fluid.param_attr.ParamAttr(
                name=name + "_weights",
                initializer=fluid.initializer.NormalInitializer(
                    loc=0.0, scale=std)),
            bias_attr=False)

        bn_name = name + "_bn"

        return fluid.layers.batch_norm(
            input=conv,
            act=act,
            is_test=(not self.is_training),
            param_attr=fluid.param_attr.ParamAttr(
                name=bn_name + "_scale",
                initializer=fluid.initializer.Constant(value=1.)),
            bias_attr=fluid.param_attr.ParamAttr(
                name=bn_name + '_offset',
                initializer=fluid.initializer.Constant(value=0.)),
            moving_mean_name=bn_name + "_mean",
            moving_variance_name=bn_name + '_variance')

    def shortcut(self, input, ch_out, stride, name):
        ch_in = input.shape[1]
        if ch_in != ch_out or stride != 1:
            filter_size = 3
            if stride == 1:
                filter_size = 1
            return self.conv_bn_layer(
                input, ch_out, filter_size, stride, name='conv' + name + '_prj')
        else:
            return input

    def bottleneck_block(self, input, num_filters, stride, cardinality,
                         reduction_ratio, name):
        conv0 = self.conv_bn_layer(
            input=input,
            num_filters=num_filters,
            filter_size=1,
            act='relu',
            name='conv' + name + '_x1')
        conv1 = self.conv_bn_layer(
            input=conv0,
            num_filters=num_filters,
            filter_size=3,
            stride=stride,
            groups=cardinality,
            act='relu',
            name='conv' + name + '_x2')
        conv2 = self.conv_bn_layer(
            input=conv1,
            num_filters=num_filters,
            filter_size=1,
            act=None,
            name='conv' + name + '_x3')
        scale = self.squeeze_excitation(
            input=conv2,
            num_channels=num_filters,
            reduction_ratio=reduction_ratio,
            name='fc' + name)
        short = self.shortcut(input, num_filters, stride, name=name)

        return fluid.layers.elementwise_add(x=short, y=scale, act='relu')

    def squeeze_excitation(self,
                           input,
                           num_channels,
                           reduction_ratio,
                           name=None):
        pool = fluid.layers.pool2d(
            input=input, pool_size=0, pool_type='avg', global_pooling=True)

        squeeze = fluid.layers.fc(
            input=pool,
            size=num_channels // reduction_ratio,
            act=None,
            param_attr=fluid.param_attr.ParamAttr(name=name + '_sqz_weights'),
            bias_attr=fluid.param_attr.ParamAttr(name=name + '_sqz_offset'))

        excitation = fluid.layers.fc(input=squeeze,
                                     size=num_channels,
                                     act='sigmoid',
                                     param_attr=fluid.param_attr.ParamAttr(
                                         name=name + '_exc_weights'),
                                     bias_attr=fluid.param_attr.ParamAttr(
                                         name=name + '_exc_offset', ))
        scale = fluid.layers.elementwise_mul(x=input, y=excitation, axis=0)

        return scale

    def net(self, input, class_dim=101):
        layers = self.layers
        seg_num = self.seg_num
        supported_layers = [50, 101, 152]
        assert layers in supported_layers, \
            "supported layers are {} but input layer is {}".format(supported_layers, layers)
        channels = input.shape[2]
        short_size = input.shape[3]
        input = fluid.layers.reshape(
            x=input, shape=[-1, channels, short_size, short_size])

        if layers == 50:
            cardinality = 32
            reduction_ratio = 16
            depth = [3, 4, 6, 3]
            num_filters = [128, 256, 512, 1024]

            conv = self.conv_bn_layer(
                input=input,
                num_filters=64,
                filter_size=7,
                stride=2,
                act='relu',
                name='conv1', )
            conv = fluid.layers.pool2d(
                input=conv,
                pool_size=3,
                pool_stride=2,
                pool_padding=1,
                pool_type='max')

        elif layers == 101:
            cardinality = 32
            reduction_ratio = 16
            depth = [3, 4, 23, 3]
            num_filters = [128, 256, 512, 1024]

            conv = self.conv_bn_layer(
                input=input,
                num_filters=64,
                filter_size=7,
                stride=2,
                act='relu',
                name="conv1", )
            conv = fluid.layers.pool2d(
                input=conv,
                pool_size=3,
                pool_stride=2,
                pool_padding=1,
                pool_type='max')

        elif layers == 152:
            cardinality = 64
            reduction_ratio = 16
            depth = [3, 8, 36, 3]
            num_filters = [256, 512, 1024, 2048]

            conv = self.conv_bn_layer(
                input=input,
                num_filters=64,
                filter_size=3,
                stride=2,
                act='relu',
                name='conv1')
            conv = self.conv_bn_layer(
                input=conv,
                num_filters=64,
                filter_size=3,
                stride=1,
                act='relu',
                name='conv2')
            conv = self.conv_bn_layer(
                input=conv,
                num_filters=128,
                filter_size=3,
                stride=1,
                act='relu',
                name='conv3')
            conv = fluid.layers.pool2d(
                input=conv, pool_size=3, pool_stride=2, pool_padding=1, \
                pool_type='max')

        n = 1 if layers == 50 or layers == 101 else 3
        for block in range(len(depth)):
            n += 1
            for i in range(depth[block]):
                conv = self.bottleneck_block(
                    input=conv,
                    num_filters=num_filters[block],
                    stride=2 if i == 0 and block != 0 else 1,
                    cardinality=cardinality,
                    reduction_ratio=reduction_ratio,
                    name=str(n) + '_' + str(i + 1))

        pool = fluid.layers.pool2d(
            input=conv, pool_size=7, pool_type='avg', global_pooling=True)
        drop = fluid.layers.dropout(x=pool, dropout_prob=0.5)

        feature = fluid.layers.reshape(
            x=drop, shape=[-1, seg_num, drop.shape[1]])
        out = fluid.layers.reduce_mean(feature, dim=1)

        stdv = 1.0 / math.sqrt(pool.shape[1] * 1.0)
        out = fluid.layers.fc(
            input=out,
            size=class_dim,
            act='softmax',
            param_attr=fluid.param_attr.ParamAttr(name='fc_weights'),
            bias_attr=fluid.param_attr.ParamAttr(name='fc_offset'))
        return out
