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

__all__ = [
    "ResNeXt101_32x8d_wsl", "ResNeXt101_32x16d_wsl", "ResNeXt101_32x32d_wsl",
    "ResNeXt101_32x48d_wsl", "Fix_ResNeXt101_32x48d_wsl"
]


class ResNeXt101_wsl():
    def __init__(self, layers=101, cardinality=32, width=48):
        self.layers = layers
        self.cardinality = cardinality
        self.width = width

    def net(self, input, class_dim=1000):
        layers = self.layers
        cardinality = self.cardinality
        width = self.width

        depth = [3, 4, 23, 3]
        base_width = cardinality * width
        num_filters = [base_width * i for i in [1, 2, 4, 8]]

        conv = self.conv_bn_layer(
            input=input,
            num_filters=64,
            filter_size=7,
            stride=2,
            act='relu',
            name="conv1")  #debug
        conv = fluid.layers.pool2d(
            input=conv,
            pool_size=3,
            pool_stride=2,
            pool_padding=1,
            pool_type='max')

        for block in range(len(depth)):
            for i in range(depth[block]):
                conv_name = 'layer' + str(block + 1) + "." + str(i)
                conv = self.bottleneck_block(
                    input=conv,
                    num_filters=num_filters[block],
                    stride=2 if i == 0 and block != 0 else 1,
                    cardinality=cardinality,
                    name=conv_name)

        pool = fluid.layers.pool2d(
            input=conv, pool_type='avg', global_pooling=True)
        stdv = 1.0 / math.sqrt(pool.shape[1] * 1.0)
        out = fluid.layers.fc(
            input=pool,
            size=class_dim,
            param_attr=fluid.param_attr.ParamAttr(
                initializer=fluid.initializer.Uniform(-stdv, stdv),
                name='fc.weight'),
            bias_attr=fluid.param_attr.ParamAttr(name='fc.bias'))
        return out

    def conv_bn_layer(self,
                      input,
                      num_filters,
                      filter_size,
                      stride=1,
                      groups=1,
                      act=None,
                      name=None):
        if "downsample" in name:
            conv_name = name + '.0'
        else:
            conv_name = name
        conv = fluid.layers.conv2d(
            input=input,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=groups,
            act=None,
            param_attr=ParamAttr(name=conv_name + ".weight"),
            bias_attr=False)
        if "downsample" in name:
            bn_name = name[:9] + 'downsample' + '.1'
        else:
            if "conv1" == name:
                bn_name = 'bn' + name[-1]
            else:
                bn_name = (name[:10] if name[7:9].isdigit() else name[:9]
                           ) + 'bn' + name[-1]
        return fluid.layers.batch_norm(
            input=conv,
            act=act,
            param_attr=ParamAttr(name=bn_name + '.weight'),
            bias_attr=ParamAttr(bn_name + '.bias'),
            moving_mean_name=bn_name + '.running_mean',
            moving_variance_name=bn_name + '.running_var', )

    def shortcut(self, input, ch_out, stride, name):
        ch_in = input.shape[1]
        if ch_in != ch_out or stride != 1:
            return self.conv_bn_layer(input, ch_out, 1, stride, name=name)
        else:
            return input

    def bottleneck_block(self, input, num_filters, stride, cardinality, name):
        cardinality = self.cardinality
        width = self.width
        conv0 = self.conv_bn_layer(
            input=input,
            num_filters=num_filters,
            filter_size=1,
            act='relu',
            name=name + ".conv1")
        conv1 = self.conv_bn_layer(
            input=conv0,
            num_filters=num_filters,
            filter_size=3,
            stride=stride,
            groups=cardinality,
            act='relu',
            name=name + ".conv2")
        conv2 = self.conv_bn_layer(
            input=conv1,
            num_filters=num_filters // (width // 8),
            filter_size=1,
            act=None,
            name=name + ".conv3")

        short = self.shortcut(
            input,
            num_filters // (width // 8),
            stride,
            name=name + ".downsample")

        return fluid.layers.elementwise_add(x=short, y=conv2, act='relu')


def ResNeXt101_32x8d_wsl():
    model = ResNeXt101_wsl(cardinality=32, width=8)
    return model


def ResNeXt101_32x16d_wsl():
    model = ResNeXt101_wsl(cardinality=32, width=16)
    return model


def ResNeXt101_32x32d_wsl():
    model = ResNeXt101_wsl(cardinality=32, width=32)
    return model


def ResNeXt101_32x48d_wsl():
    model = ResNeXt101_wsl(cardinality=32, width=48)
    return model


def Fix_ResNeXt101_32x48d_wsl():
    model = ResNeXt101_wsl(cardinality=32, width=48)
    return model
