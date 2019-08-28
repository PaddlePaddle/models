#copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
import paddle.fluid as fluid
import math
from paddle.fluid.param_attr import ParamAttr

__all__ = ["SqueezeNet", "SqueezeNet1_0", "SqueezeNet1_1"]


class SqueezeNet():
    def __init__(self, version='1.0'):
        self.version = version

    def net(self, input, class_dim=1000):
        version = self.version
        assert version in ['1.0', '1.1'], \
            "supported version are {} but input version is {}".format(['1.0', '1.1'], version)
        if version == '1.0':
            conv = fluid.layers.conv2d(
                input,
                num_filters=96,
                filter_size=7,
                stride=2,
                act='relu',
                param_attr=fluid.param_attr.ParamAttr(name="conv1_weights"),
                bias_attr=ParamAttr(name='conv1_offset'))
            conv = fluid.layers.pool2d(
                conv, pool_size=3, pool_stride=2, pool_type='max')
            conv = self.make_fire(conv, 16, 64, 64, name='fire2')
            conv = self.make_fire(conv, 16, 64, 64, name='fire3')
            conv = self.make_fire(conv, 32, 128, 128, name='fire4')
            conv = fluid.layers.pool2d(
                conv, pool_size=3, pool_stride=2, pool_type='max')
            conv = self.make_fire(conv, 32, 128, 128, name='fire5')
            conv = self.make_fire(conv, 48, 192, 192, name='fire6')
            conv = self.make_fire(conv, 48, 192, 192, name='fire7')
            conv = self.make_fire(conv, 64, 256, 256, name='fire8')
            conv = fluid.layers.pool2d(
                conv, pool_size=3, pool_stride=2, pool_type='max')
            conv = self.make_fire(conv, 64, 256, 256, name='fire9')
        else:
            conv = fluid.layers.conv2d(
                input,
                num_filters=64,
                filter_size=3,
                stride=2,
                padding=1,
                act='relu',
                param_attr=fluid.param_attr.ParamAttr(name="conv1_weights"),
                bias_attr=ParamAttr(name='conv1_offset'))
            conv = fluid.layers.pool2d(
                conv, pool_size=3, pool_stride=2, pool_type='max')
            conv = self.make_fire(conv, 16, 64, 64, name='fire2')
            conv = self.make_fire(conv, 16, 64, 64, name='fire3')
            conv = fluid.layers.pool2d(
                conv, pool_size=3, pool_stride=2, pool_type='max')
            conv = self.make_fire(conv, 32, 128, 128, name='fire4')
            conv = self.make_fire(conv, 32, 128, 128, name='fire5')
            conv = fluid.layers.pool2d(
                conv, pool_size=3, pool_stride=2, pool_type='max')
            conv = self.make_fire(conv, 48, 192, 192, name='fire6')
            conv = self.make_fire(conv, 48, 192, 192, name='fire7')
            conv = self.make_fire(conv, 64, 256, 256, name='fire8')
            conv = self.make_fire(conv, 64, 256, 256, name='fire9')
        conv = fluid.layers.dropout(conv, dropout_prob=0.5)
        conv = fluid.layers.conv2d(
            conv,
            num_filters=class_dim,
            filter_size=1,
            act='relu',
            param_attr=fluid.param_attr.ParamAttr(name="conv10_weights"),
            bias_attr=ParamAttr(name='conv10_offset'))
        conv = fluid.layers.pool2d(conv, pool_type='avg', global_pooling=True)
        out = fluid.layers.flatten(conv)
        return out

    def make_fire_conv(self,
                       input,
                       num_filters,
                       filter_size,
                       padding=0,
                       name=None):
        conv = fluid.layers.conv2d(
            input,
            num_filters=num_filters,
            filter_size=filter_size,
            padding=padding,
            act='relu',
            param_attr=fluid.param_attr.ParamAttr(name=name + "_weights"),
            bias_attr=ParamAttr(name=name + '_offset'))
        return conv

    def make_fire(self,
                  input,
                  squeeze_channels,
                  expand1x1_channels,
                  expand3x3_channels,
                  name=None):
        conv = self.make_fire_conv(
            input, squeeze_channels, 1, name=name + '_squeeze1x1')
        conv_path1 = self.make_fire_conv(
            conv, expand1x1_channels, 1, name=name + '_expand1x1')
        conv_path2 = self.make_fire_conv(
            conv, expand3x3_channels, 3, 1, name=name + '_expand3x3')
        out = fluid.layers.concat([conv_path1, conv_path2], axis=1)
        return out


def SqueezeNet1_0():
    model = SqueezeNet(version='1.0')
    return model


def SqueezeNet1_1():
    model = SqueezeNet(version='1.1')
    return model
