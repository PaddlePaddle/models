#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import functools
import numpy as np
import time
import os
import math

import paddle
import paddle.fluid as fluid
import paddle.fluid.profiler as profiler
import utils

__all__ = ["FastImageNet"]


class FastImageNet():
    def __init__(self, layers=50, is_train=True):
        self.layers = layers
        self.is_train = is_train

    def net(self, input, class_dim=1000, img_size=224, is_train=True):
        layers = self.layers
        supported_layers = [50, 101, 152]
        assert layers in supported_layers, \
            "supported layers are {} but input layer is {}".format(supported_layers, layers)

        if layers == 50:
            depth = [3, 4, 6, 3]
        elif layers == 101:
            depth = [3, 4, 23, 3]
        elif layers == 152:
            depth = [3, 8, 36, 3]
        num_filters = [64, 128, 256, 512]

        conv = self.conv_bn_layer(
            input=input, num_filters=64, filter_size=7, stride=2, act='relu')
        conv = fluid.layers.pool2d(
            input=conv,
            pool_size=3,
            pool_stride=2,
            pool_padding=1,
            pool_type='max')

        for block in range(len(depth)):
            for i in range(depth[block]):
                conv = self.bottleneck_block(
                    input=conv,
                    num_filters=num_filters[block],
                    stride=2 if i == 0 and block != 0 else 1)
        pool_size = int(img_size / 32)
        pool = fluid.layers.pool2d(
            input=conv, pool_type='avg', global_pooling=True)
        out = fluid.layers.fc(
            input=pool,
            size=class_dim,
            act=None,
            param_attr=fluid.param_attr.ParamAttr(
                initializer=fluid.initializer.NormalInitializer(0.0, 0.01),
                regularizer=fluid.regularizer.L2Decay(1e-4)),
            bias_attr=fluid.ParamAttr(
                regularizer=fluid.regularizer.L2Decay(1e-4)))
        return out

    def conv_bn_layer(self,
                      input,
                      num_filters,
                      filter_size,
                      stride=1,
                      groups=1,
                      act=None,
                      bn_init_value=1.0):
        conv = fluid.layers.conv2d(
            input=input,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=groups,
            act=None,
            bias_attr=False,
            param_attr=fluid.ParamAttr(
                regularizer=fluid.regularizer.L2Decay(1e-4)))
        return fluid.layers.batch_norm(
            input=conv,
            act=act,
            is_test=not self.is_train,
            param_attr=fluid.param_attr.ParamAttr(
                initializer=fluid.initializer.Constant(bn_init_value),
                regularizer=None))

    def shortcut(self, input, ch_out, stride):
        ch_in = input.shape[1]
        if ch_in != ch_out or stride != 1:
            return self.conv_bn_layer(input, ch_out, 1, stride)
        else:
            return input

    def bottleneck_block(self, input, num_filters, stride):
        conv0 = self.conv_bn_layer(
            input=input, num_filters=num_filters, filter_size=1, act='relu')
        conv1 = self.conv_bn_layer(
            input=conv0,
            num_filters=num_filters,
            filter_size=3,
            stride=stride,
            act='relu')
        # init bn-weight0
        conv2 = self.conv_bn_layer(
            input=conv1,
            num_filters=num_filters * 4,
            filter_size=1,
            act=None,
            bn_init_value=0.0)

        short = self.shortcut(input, num_filters * 4, stride)

        return fluid.layers.elementwise_add(x=short, y=conv2, act='relu')


def lr_decay(lrs, epochs, bs, total_image):
    boundaries = []
    values = []
    for idx, epoch in enumerate(epochs):
        step = total_image // bs[idx]
        if step * bs[idx] < total_image:
            step += 1
        ratio = (lrs[idx][1] - lrs[idx][0]) * 1.0 / (epoch[1] - epoch[0])
        lr_base = lrs[idx][0]
        for s in range(epoch[0], epoch[1]):
            if boundaries:
                boundaries.append(boundaries[-1] + step + 1)
            else:
                boundaries = [step]
            lr = lr_base + ratio * (s - epoch[0])
            values.append(lr)
            print("epoch: [%d], steps: [%d], lr: [%f]" %
                  (s, boundaries[-1], values[-1]))
    values.append(lrs[-1])
    print("epoch: [%d:], steps: [%d:], lr:[%f]" %
          (epochs[-1][-1], boundaries[-1], values[-1]))
    return boundaries, values
