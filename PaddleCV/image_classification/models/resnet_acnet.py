#copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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

import math

import paddle
import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr

__all__ = [
    "ResNet_ACNet", "ResNet18_ACNet", "ResNet34_ACNet", "ResNet50_ACNet",
    "ResNet101_ACNet", "ResNet152_ACNet"
]


class ResNetACNet(object):
    """ ACNet """

    def __init__(self, layers=50, deploy=False):
        """init"""
        self.layers = layers
        self.deploy = deploy

    def net(self, input, class_dim=1000):
        """model"""
        layers = self.layers
        supported_layers = [18, 34, 50, 101, 152]
        assert layers in supported_layers, \
            "supported layers are {} but input layer is {}".format(supported_layers, layers)

        if layers == 18:
            depth = [2, 2, 2, 2]
        elif layers == 34 or layers == 50:
            depth = [3, 4, 6, 3]
        elif layers == 101:
            depth = [3, 4, 23, 3]
        elif layers == 152:
            depth = [3, 8, 36, 3]
        num_filters = [64, 128, 256, 512]

        conv = self.conv_bn_layer(
            input=input,
            num_filters=64,
            filter_size=7,
            stride=2,
            act='relu',
            name="conv1")
        conv = fluid.layers.pool2d(
            input=conv,
            pool_size=3,
            pool_stride=2,
            pool_padding=1,
            pool_type='max')
        if layers >= 50:
            for block in range(len(depth)):
                for i in range(depth[block]):
                    if layers in [101, 152] and block == 2:
                        if i == 0:
                            conv_name = "res" + str(block + 2) + "a"
                        else:
                            conv_name = "res" + str(block + 2) + "b" + str(i)
                    else:
                        conv_name = "res" + str(block + 2) + chr(97 + i)
                    conv = self.bottleneck_block(
                        input=conv,
                        num_filters=num_filters[block],
                        stride=2 if i == 0 and block != 0 else 1,
                        name=conv_name)
        else:
            for block in range(len(depth)):
                for i in range(depth[block]):
                    conv_name = "res" + str(block + 2) + chr(97 + i)
                    conv = self.basic_block(
                        input=conv,
                        num_filters=num_filters[block],
                        stride=2 if i == 0 and block != 0 else 1,
                        is_first=block == i == 0,
                        name=conv_name)

        pool = fluid.layers.pool2d(
            input=conv, pool_size=7, pool_type='avg', global_pooling=True)

        stdv = 1.0 / math.sqrt(pool.shape[1] * 1.0)
        out = fluid.layers.fc(
            input=pool,
            size=class_dim,
            param_attr=fluid.param_attr.ParamAttr(
                initializer=fluid.initializer.Uniform(-stdv, stdv)))
        return out

    def conv_bn_layer(self, **kwargs):
        """
        conv_bn_layer
        """
        if kwargs['filter_size'] == 1:
            return self.conv_bn_layer_ori(**kwargs)
        else:
            return self.conv_bn_layer_ac(**kwargs)

    # conv bn+relu
    def conv_bn_layer_ori(self,
                          input,
                          num_filters,
                          filter_size,
                          stride=1,
                          groups=1,
                          act=None,
                          name=None):
        """
        standard convbn
        used for 1x1 convbn in acnet
        """
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
            name=name + '.conv2d.output.1')

        if name == "conv1":
            bn_name = "bn_" + name
        else:
            bn_name = "bn" + name[3:]
        return fluid.layers.batch_norm(
            input=conv,
            act=act,
            name=bn_name + '.output.1',
            param_attr=ParamAttr(name=bn_name + '_scale'),
            bias_attr=ParamAttr(bn_name + '_offset'),
            moving_mean_name=bn_name + '_mean',
            moving_variance_name=bn_name + '_variance', )

    # conv bn+relu
    def conv_bn_layer_ac(self,
                         input,
                         num_filters,
                         filter_size,
                         stride=1,
                         groups=1,
                         act=None,
                         name=None):
        """ ACNet conv bn """
        padding = (filter_size - 1) // 2

        square_conv = fluid.layers.conv2d(
            input=input,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            groups=groups,
            act=act if self.deploy else None,
            param_attr=ParamAttr(name=name + "_acsquare_weights"),
            bias_attr=ParamAttr(name=name + "_acsquare_bias")
            if self.deploy else False,
            name=name + '.acsquare.conv2d.output.1')

        if self.deploy:
            return square_conv
        else:
            ver_conv = fluid.layers.conv2d(
                input=input,
                num_filters=num_filters,
                filter_size=(filter_size, 1),
                stride=stride,
                padding=(padding, 0),
                groups=groups,
                act=None,
                param_attr=ParamAttr(name=name + "_acver_weights"),
                bias_attr=False,
                name=name + '.acver.conv2d.output.1')

            hor_conv = fluid.layers.conv2d(
                input=input,
                num_filters=num_filters,
                filter_size=(1, filter_size),
                stride=stride,
                padding=(0, padding),
                groups=groups,
                act=None,
                param_attr=ParamAttr(name=name + "_achor_weights"),
                bias_attr=False,
                name=name + '.achor.conv2d.output.1')

            if name == "conv1":
                bn_name = "bn_" + name
            else:
                bn_name = "bn" + name[3:]

            square_bn = fluid.layers.batch_norm(
                input=square_conv,
                act=None,
                name=bn_name + '.acsquare.output.1',
                param_attr=ParamAttr(name=bn_name + '_acsquare_scale'),
                bias_attr=ParamAttr(bn_name + '_acsquare_offset'),
                moving_mean_name=bn_name + '_acsquare_mean',
                moving_variance_name=bn_name + '_acsquare_variance', )

            ver_bn = fluid.layers.batch_norm(
                input=ver_conv,
                act=None,
                name=bn_name + '.acver.output.1',
                param_attr=ParamAttr(name=bn_name + '_acver_scale'),
                bias_attr=ParamAttr(bn_name + '_acver_offset'),
                moving_mean_name=bn_name + '_acver_mean',
                moving_variance_name=bn_name + '_acver_variance', )

            hor_bn = fluid.layers.batch_norm(
                input=hor_conv,
                act=None,
                name=bn_name + '.achor.output.1',
                param_attr=ParamAttr(name=bn_name + '_achor_scale'),
                bias_attr=ParamAttr(bn_name + '_achor_offset'),
                moving_mean_name=bn_name + '_achor_mean',
                moving_variance_name=bn_name + '_achor_variance', )

            return fluid.layers.elementwise_add(
                x=square_bn, y=ver_bn + hor_bn, act=act)

    def shortcut(self, input, ch_out, stride, is_first, name):
        """ shortcut """
        ch_in = input.shape[1]
        if ch_in != ch_out or stride != 1 or is_first == True:
            return self.conv_bn_layer(
                input=input,
                num_filters=ch_out,
                filter_size=1,
                stride=stride,
                name=name)
        else:
            return input

    def bottleneck_block(self, input, num_filters, stride, name):
        """" bottleneck_block """
        conv0 = self.conv_bn_layer(
            input=input,
            num_filters=num_filters,
            filter_size=1,
            act='relu',
            name=name + "_branch2a")
        conv1 = self.conv_bn_layer(
            input=conv0,
            num_filters=num_filters,
            filter_size=3,
            stride=stride,
            act='relu',
            name=name + "_branch2b")
        conv2 = self.conv_bn_layer(
            input=conv1,
            num_filters=num_filters * 4,
            filter_size=1,
            act=None,
            name=name + "_branch2c")

        short = self.shortcut(
            input,
            num_filters * 4,
            stride,
            is_first=False,
            name=name + "_branch1")

        return fluid.layers.elementwise_add(
            x=short, y=conv2, act='relu', name=name + ".add.output.5")

    def basic_block(self, input, num_filters, stride, is_first, name):
        """ basic_block """
        conv0 = self.conv_bn_layer(
            input=input,
            num_filters=num_filters,
            filter_size=3,
            act='relu',
            stride=stride,
            name=name + "_branch2a")
        conv1 = self.conv_bn_layer(
            input=conv0,
            num_filters=num_filters,
            filter_size=3,
            act=None,
            name=name + "_branch2b")
        short = self.shortcut(
            input, num_filters, stride, is_first, name=name + "_branch1")
        return fluid.layers.elementwise_add(x=short, y=conv1, act='relu')


def ResNet18_ACNet(deploy=False):
    """ResNet18 + ACNet"""
    model = ResNet_ACNet(layers=18, deploy=deploy)
    return model


def ResNet34_ACNet(deploy=False):
    """ResNet34 + ACNet"""
    model = ResNetACNet(layers=34, deploy=deploy)
    return model


def ResNet50_ACNet(deploy=False):
    """ResNet50 + ACNet"""
    model = ResNetACNet(layers=50, deploy=deploy)
    return model


def ResNet101_ACNet(deploy=False):
    """ResNet101 + ACNet"""
    model = ResNetACNet(layers=101, deploy=deploy)
    return model


def ResNet152_ACNet(deploy=False):
    """ResNet152 + ACNet"""
    model = ResNetACNet(layers=152, deploy=deploy)
    return model
