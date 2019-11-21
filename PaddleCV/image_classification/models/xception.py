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
import sys
from paddle.fluid.param_attr import ParamAttr

__all__ = ['Xception', 'Xception41', 'Xception65', 'Xception71']


class Xception(object):
    """Xception"""

    def __init__(self, entry_flow_block_num=3, middle_flow_block_num=8):
        self.entry_flow_block_num = entry_flow_block_num
        self.middle_flow_block_num = middle_flow_block_num
        return

    def net(self, input, class_dim=1000):
        conv = self.entry_flow(input, self.entry_flow_block_num)
        conv = self.middle_flow(conv, self.middle_flow_block_num)
        conv = self.exit_flow(conv, class_dim)

        return conv

    def entry_flow(self, input, block_num=3):
        '''xception entry_flow'''
        name = "entry_flow"
        conv = self.conv_bn_layer(
            input=input,
            num_filters=32,
            filter_size=3,
            stride=2,
            act='relu',
            name=name + "_conv1")
        conv = self.conv_bn_layer(
            input=conv,
            num_filters=64,
            filter_size=3,
            stride=1,
            act='relu',
            name=name + "_conv2")

        if block_num == 3:
            relu_first = [False, True, True]
            num_filters = [128, 256, 728]
            stride = [2, 2, 2]
        elif block_num == 5:
            relu_first = [False, True, True, True, True]
            num_filters = [128, 256, 256, 728, 728]
            stride = [2, 1, 2, 1, 2]
        else:
            sys.exit(-1)

        for block in range(block_num):
            curr_name = "{}_{}".format(name, block)
            conv = self.entry_flow_bottleneck_block(
                conv,
                num_filters=num_filters[block],
                name=curr_name,
                stride=stride[block],
                relu_first=relu_first[block])

        return conv

    def entry_flow_bottleneck_block(self,
                                    input,
                                    num_filters,
                                    name,
                                    stride=2,
                                    relu_first=False):
        '''entry_flow_bottleneck_block'''
        short = fluid.layers.conv2d(
            input=input,
            num_filters=num_filters,
            filter_size=1,
            stride=stride,
            padding=0,
            act=None,
            param_attr=ParamAttr(name + "_branch1_weights"),
            bias_attr=False)

        conv0 = input
        if relu_first:
            conv0 = fluid.layers.relu(conv0)

        conv1 = self.separable_conv(
            conv0, num_filters, stride=1, name=name + "_branch2a_weights")

        conv2 = fluid.layers.relu(conv1)
        conv2 = self.separable_conv(
            conv2, num_filters, stride=1, name=name + "_branch2b_weights")

        pool = fluid.layers.pool2d(
            input=conv2,
            pool_size=3,
            pool_stride=stride,
            pool_padding=1,
            pool_type='max')

        return fluid.layers.elementwise_add(x=short, y=pool)

    def middle_flow(self, input, block_num=8):
        '''xception middle_flow'''
        num_filters = 728
        conv = input
        for block in range(block_num):
            name = "middle_flow_{}".format(block)
            conv = self.middle_flow_bottleneck_block(conv, num_filters, name)

        return conv

    def middle_flow_bottleneck_block(self, input, num_filters, name):
        '''middle_flow_bottleneck_block'''
        conv0 = fluid.layers.relu(input)
        conv0 = self.separable_conv(
            conv0,
            num_filters=num_filters,
            stride=1,
            name=name + "_branch2a_weights")

        conv1 = fluid.layers.relu(conv0)
        conv1 = self.separable_conv(
            conv1,
            num_filters=num_filters,
            stride=1,
            name=name + "_branch2b_weights")

        conv2 = fluid.layers.relu(conv1)
        conv2 = self.separable_conv(
            conv2,
            num_filters=num_filters,
            stride=1,
            name=name + "_branch2c_weights")

        return fluid.layers.elementwise_add(x=input, y=conv2)

    def exit_flow(self, input, class_dim):
        '''xception exit flow'''
        name = "exit_flow"
        num_filters1 = 728
        num_filters2 = 1024
        conv0 = self.exit_flow_bottleneck_block(
            input, num_filters1, num_filters2, name=name + "_1")

        conv1 = self.separable_conv(
            conv0, num_filters=1536, stride=1, name=name + "_2")
        conv1 = fluid.layers.relu(conv1)

        conv2 = self.separable_conv(
            conv1, num_filters=2048, stride=1, name=name + "_3")
        conv2 = fluid.layers.relu(conv2)

        pool = fluid.layers.pool2d(
            input=conv2, pool_type='avg', global_pooling=True)

        stdv = 1.0 / math.sqrt(pool.shape[1] * 1.0)
        out = fluid.layers.fc(
            input=pool,
            size=class_dim,
            param_attr=fluid.param_attr.ParamAttr(
                name='fc_weights',
                initializer=fluid.initializer.Uniform(-stdv, stdv)),
            bias_attr=fluid.param_attr.ParamAttr(name='fc_offset'))

        return out

    def exit_flow_bottleneck_block(self, input, num_filters1, num_filters2,
                                   name):
        '''entry_flow_bottleneck_block'''
        short = fluid.layers.conv2d(
            input=input,
            num_filters=num_filters2,
            filter_size=1,
            stride=2,
            padding=0,
            act=None,
            param_attr=ParamAttr(name + "_branch1_weights"),
            bias_attr=False)

        conv0 = fluid.layers.relu(input)
        conv1 = self.separable_conv(
            conv0, num_filters1, stride=1, name=name + "_branch2a_weights")

        conv2 = fluid.layers.relu(conv1)
        conv2 = self.separable_conv(
            conv2, num_filters2, stride=1, name=name + "_branch2b_weights")

        pool = fluid.layers.pool2d(
            input=conv2,
            pool_size=3,
            pool_stride=2,
            pool_padding=1,
            pool_type='max')

        return fluid.layers.elementwise_add(x=short, y=pool)

    def separable_conv(self, input, num_filters, stride=1, name=None):
        """separable_conv"""
        pointwise_conv = self.conv_bn_layer(
            input=input,
            filter_size=1,
            num_filters=num_filters,
            stride=1,
            name=name + "_sep")

        depthwise_conv = self.conv_bn_layer(
            input=pointwise_conv,
            filter_size=3,
            num_filters=num_filters,
            stride=stride,
            groups=num_filters,
            use_cudnn=False,
            name=name + "_dw")

        return depthwise_conv

    def conv_bn_layer(self,
                      input,
                      num_filters,
                      filter_size,
                      stride=1,
                      groups=1,
                      act=None,
                      use_cudnn=True,
                      name=None):
        """conv_bn_layer"""
        conv = fluid.layers.conv2d(
            input=input,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=groups,
            act=None,
            param_attr=ParamAttr(name=name + "_weights"),
            bias_attr=False,
            use_cudnn=use_cudnn)

        bn_name = "bn_" + name

        return fluid.layers.batch_norm(
            input=conv,
            act=act,
            param_attr=ParamAttr(name=bn_name + '_scale'),
            bias_attr=ParamAttr(bn_name + '_offset'),
            moving_mean_name=bn_name + '_mean',
            moving_variance_name=bn_name + '_variance')


def Xception41():
    model = Xception(entry_flow_block_num=3, middle_flow_block_num=8)
    return model


def Xception65():
    model = Xception(entry_flow_block_num=3, middle_flow_block_num=16)
    return model


def Xception71():
    model = Xception(entry_flow_block_num=5, middle_flow_block_num=16)
    return model
