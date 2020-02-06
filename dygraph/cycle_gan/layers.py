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

from __future__ import division
import paddle.fluid as fluid
import numpy as np
from paddle.fluid.dygraph.nn import Conv2D,  Conv2DTranspose , BatchNorm ,Pool2D
import os

# cudnn is not better when batch size is 1.
use_cudnn = False


class conv2d(fluid.dygraph.Layer):
    """docstring for Conv2D"""
    def __init__(self,
                num_channels,
                num_filters=64,
                filter_size=7,
                stride=1,
                stddev=0.02,
                padding=0,
                norm=True,
                relu=True,
                relufactor=0.0,
                use_bias=False):
        super(conv2d, self).__init__()

        if use_bias == False:
            con_bias_attr = False
        else:
            con_bias_attr = fluid.ParamAttr(initializer=fluid.initializer.Constant(0.0))

        self.conv = Conv2D(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            use_cudnn=use_cudnn,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.NormalInitializer(loc=0.0,scale=stddev)),
            bias_attr=con_bias_attr)
        if norm:
            self.bn = BatchNorm(
                num_channels=num_filters,
                param_attr=fluid.ParamAttr(
                    initializer=fluid.initializer.NormalInitializer(1.0,0.02)),
                bias_attr=fluid.ParamAttr(
                    initializer=fluid.initializer.Constant(0.0)),
                trainable_statistics=True
                )
    
        self.relufactor = relufactor
        self.use_bias = use_bias
        self.norm = norm
        self.relu = relu

    
    def forward(self,inputs):
        conv = self.conv(inputs)
        if self.norm:
            conv = self.bn(conv)
        if self.relu:
            conv = fluid.layers.leaky_relu(conv,alpha=self.relufactor)
        return conv


class DeConv2D(fluid.dygraph.Layer):
    def __init__(self,
            num_channels,
            num_filters=64,
            filter_size=7,
            stride=1,
            stddev=0.02,
            padding=[0,0],
            outpadding=[0,0,0,0],
            relu=True,
            norm=True,
            relufactor=0.0,
            use_bias=False
            ):
        super(DeConv2D,self).__init__()

        if use_bias == False:
            de_bias_attr = False
        else:
            de_bias_attr = fluid.ParamAttr(initializer=fluid.initializer.Constant(0.0))

        self._deconv = Conv2DTranspose(num_channels,
                                       num_filters,
                                       filter_size=filter_size,
                                       stride=stride,
                                       padding=padding,
                                       param_attr=fluid.ParamAttr(
                                           initializer=fluid.initializer.NormalInitializer(loc=0.0, scale=stddev)),
                                       bias_attr=de_bias_attr)



        if norm:
            self.bn = BatchNorm(
                num_channels=num_filters,
                param_attr=fluid.ParamAttr(
                    initializer=fluid.initializer.NormalInitializer(1.0, 0.02)),
                bias_attr=fluid.ParamAttr(initializer=fluid.initializer.Constant(0.0)),
                trainable_statistics=True)        
        self.outpadding = outpadding
        self.relufactor = relufactor
        self.use_bias = use_bias
        self.norm = norm
        self.relu = relu

    def forward(self,inputs):
        #todo: add use_bias
        #if self.use_bias==False:
        conv = self._deconv(inputs)
        #else:
        #    conv = self._deconv(inputs)
        conv = fluid.layers.pad2d(conv, paddings=self.outpadding, mode='constant', pad_value=0.0)

        if self.norm:
            conv = self.bn(conv)
        if self.relu:
            conv = fluid.layers.leaky_relu(conv,alpha=self.relufactor)
        return conv
