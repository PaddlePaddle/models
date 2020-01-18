# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, BatchNorm
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.initializer import Constant
from config import cfg
import numpy as np


class conv_bn_layer(fluid.dygraph.Layer):
    def __init__(self,
                 name_scope,
                 ch_in,
                 ch_out,
                 filter_size,
                 stride,
                 padding,
                 act='relu'):
        super(conv_bn_layer, self).__init__()

        self._conv = Conv2D(
            num_channels == ch_in,
            num_filters=ch_out,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            act=None,
            param_attr=ParamAttr(name=name_scope + "_weights"),
            bias_attr=ParamAttr(name=name_scope + "_bias"))

        if name_scope == "conv1":
            bn_name = "bn_" + name_scope
        else:
            bn_name = "bn" + name_scope[3:]

        self._bn = BatchNorm(
            bn_name + '.output.1',
            num_channels=ch_out,
            act=act,
            param_attr=ParamAttr(name=bn_name + '_scale'),
            bias_attr=ParamAttr(bn_name + '_offset'),
            moving_mean_name=bn_name + '_mean',
            moving_variance_name=bn_name + '_variance',
            is_test=True)

    def forward(self, inputs):
        conv = self._conv(inputs)
        out = self._bn(conv)

        return out


class conv_affine_layer(fluid.dygraph.Layer):
    def __init__(self,
                 name_scope,
                 ch_in,
                 ch_out,
                 filter_size,
                 stride,
                 padding,
                 learning_rate=1.0,
                 act='relu'):
        super(conv_affine_layer, self).__init__()

        self._conv = Conv2D(
            num_channels=ch_in,
            num_filters=ch_out,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            act=None,
            param_attr=ParamAttr(
                name=name_scope + "_weights", learning_rate=learning_rate),
            bias_attr=False)

        if name_scope == "conv1":
            bn_name = "bn_" + name_scope
        else:
            bn_name = "bn" + name_scope[3:]
        self.name_scope = name_scope

        self.scale = fluid.layers.create_parameter(
            shape=[ch_out],
            dtype='float32',
            attr=ParamAttr(
                name=bn_name + '_scale', learning_rate=0.),
            default_initializer=Constant(1.))
        self.bias = fluid.layers.create_parameter(
            shape=[ch_out],
            dtype='float32',
            attr=ParamAttr(
                bn_name + '_offset', learning_rate=0.),
            default_initializer=Constant(0.))

        self.act = act

    def forward(self, inputs):
        if cfg.enable_ce:
            if self.name_scope == "conv1":
                print("conv1 affine channel scale {} {}".format(
                    np.mean(np.abs(self.scale.numpy())),
                    self.scale.numpy().shape))
                print("conv1 affine channel bias {} {}".format(
                    np.mean(np.abs(self.bias.numpy())), self.bias.numpy()
                    .shape))
        conv = self._conv(inputs)
        out = fluid.layers.affine_channel(
            x=conv, scale=self.scale, bias=self.bias)
        if self.act == 'relu':
            out = fluid.layers.relu(x=out)
        return out


class bottleneck(fluid.dygraph.Layer):
    def __init__(self,
                 name_scope,
                 ch_in,
                 ch_out,
                 stride,
                 shortcut=True,
                 learning_rate=1.0):
        super(bottleneck, self).__init__()

        self.shortcut = shortcut
        if not shortcut:
            self.short = conv_affine_layer(
                name_scope + "_branch1",
                ch_in=ch_in,
                ch_out=ch_out * 4,
                filter_size=1,
                stride=stride,
                padding=0,
                act=None,
                learning_rate=learning_rate)

        self.conv1 = conv_affine_layer(
            name_scope + "_branch2a",
            ch_in=ch_in,
            ch_out=ch_out,
            filter_size=1,
            stride=stride,
            padding=0,
            learning_rate=learning_rate, )

        self.conv2 = conv_affine_layer(
            name_scope + "_branch2b",
            ch_in=ch_out,
            ch_out=ch_out,
            filter_size=3,
            stride=1,
            padding=1,
            learning_rate=learning_rate)

        self.conv3 = conv_affine_layer(
            name_scope + "_branch2c",
            ch_in=ch_out,
            ch_out=ch_out * 4,
            filter_size=1,
            stride=1,
            padding=0,
            learning_rate=learning_rate,
            act=None)
        self.name_scope = name_scope

    def forward(self, inputs):

        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)

        conv1 = self.conv1(inputs)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)

        out = fluid.layers.elementwise_add(
            x=short,
            y=conv3,
            act='relu',
            name=self.name_scope + ".add.output.5")

        return out


class bottleneck_list(fluid.dygraph.Layer):
    def __init__(self,
                 name_scope,
                 ch_in,
                 ch_out,
                 count,
                 stride,
                 learning_rate=1.0):
        super(bottleneck_list, self).__init__()

        self.bottleneck_block_list = []
        for i in range(count):
            if i == 0:
                name = name_scope + "a"
                self.stride = stride
                self.shortcut = False
            else:
                name = name_scope + chr(ord("a") + i)
                self.stride = 1
                self.shortcut = True

            bottleneck_block = self.add_sublayer(
                name,
                bottleneck(
                    name,
                    ch_in=ch_in if i == 0 else ch_out * 4,
                    ch_out=ch_out,
                    stride=self.stride,
                    shortcut=self.shortcut,
                    learning_rate=learning_rate))
            self.bottleneck_block_list.append(bottleneck_block)
            shortcut = True

    def forward(self, inputs):
        res_out = self.bottleneck_block_list[0](inputs)
        for bottleneck_block in self.bottleneck_block_list[1:]:
            res_out = bottleneck_block(res_out)
        return res_out


class add_ResNet50_conv_body(fluid.dygraph.Layer):
    def __init__(self, name_scope, layers=50):
        super(add_ResNet50_conv_body, self).__init__()

        if layers == 50:
            depth = [3, 4, 6, 3]
        elif layers == 101:
            depth = [3, 4, 23, 3]
        elif layers == 152:
            depth = [3, 8, 36, 3]
        #num_channels = [64, 256, 512, 1024]
        #num_filters = [64, 128, 256, 512]
        self.conv = conv_affine_layer(
            "conv1",
            ch_in=3,
            ch_out=64,
            filter_size=7,
            stride=2,
            padding=3,
            learning_rate=0.)

        self.pool2d_max = Pool2D(
            pool_type='max', pool_size=3, pool_stride=2, pool_padding=1)

        self.stage2 = bottleneck_list(
            "res2",
            ch_in=64,
            ch_out=64,
            count=depth[0],
            stride=1,
            learning_rate=0.)

        self.stage3 = bottleneck_list(
            "res3", ch_in=256, ch_out=128, count=depth[1], stride=2)

        self.stage4 = bottleneck_list(
            "res4", ch_in=512, ch_out=256, count=depth[2], stride=2)

    def forward(self, inputs):
        conv1 = self.conv(inputs)
        if cfg.enable_ce:
            print('conv1 {} {}'.format(abs(conv1.numpy()).sum(), conv1.shape))
        poo1 = self.pool2d_max(conv1)
        outs = []

        res2 = self.stage2(poo1)
        if cfg.enable_ce:
            print('res2: {}'.format(np.abs(res2.numpy()).mean()), res2.shape)
        if cfg.TRAIN.freeze_at == 2:
            res2.stop_gradient = True
        outs.append(res2)

        res3 = self.stage3(res2)
        if cfg.enable_ce:
            print('res3: {}'.format(np.abs(res3.numpy()).mean()), res3.shape)
        outs.append(res3)
        if cfg.TRAIN.freeze_at == 3:
            res3.stop_gradient = True

        res4 = self.stage4(res3)
        if cfg.enable_ce:
            print('res4: {}'.format(np.abs(res4.numpy()).mean()), res4.shape)
        outs.append(res4)
        if cfg.TRAIN.freeze_at == 4:
            res4.stop_gradient = True

        return tuple(outs)
