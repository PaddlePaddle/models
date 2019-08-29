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

from .base_network import conv2d, deconv2d, norm_layer
import paddle.fluid as fluid
import numpy as np


class StarGAN_model(object):
    def __init__(self):
        pass

    def ResidualBlock(self, input, dim, name):
        conv0 = conv2d(
            input,
            num_filters=dim,
            filter_size=3,
            stride=1,
            padding=1,
            use_bias=False,
            norm="instance_norm",
            activation_fn='relu',
            name=name + ".main0",
            initial='kaiming')
        conv1 = conv2d(
            input=conv0,
            num_filters=dim,
            filter_size=3,
            stride=1,
            padding=1,
            use_bias=False,
            norm="instance_norm",
            activation_fn=None,
            name=name + ".main3",
            initial='kaiming')
        return input + conv1

    def network_G(self, input, label_trg, cfg, name="generator"):
        repeat_num = 6
        shape = input.shape
        label_trg_e = fluid.layers.reshape(label_trg,
                                           [-1, label_trg.shape[1], 1, 1])
        label_trg_e = fluid.layers.expand(
            x=label_trg_e, expand_times=[1, 1, shape[2], shape[3]])
        input1 = fluid.layers.concat([input, label_trg_e], 1)
        conv0 = conv2d(
            input=input1,
            num_filters=cfg.g_base_dims,
            filter_size=7,
            stride=1,
            padding=3,
            use_bias=False,
            norm="instance_norm",
            activation_fn='relu',
            name=name + '0',
            initial='kaiming')
        conv_down = conv0
        for i in range(2):
            rate = 2**(i + 1)
            conv_down = conv2d(
                input=conv_down,
                num_filters=cfg.g_base_dims * rate,
                filter_size=4,
                stride=2,
                padding=1,
                use_bias=False,
                norm="instance_norm",
                activation_fn='relu',
                name=name + str(i * 3 + 3),
                initial='kaiming')
        res_block = conv_down
        for i in range(repeat_num):
            res_block = self.ResidualBlock(
                res_block,
                cfg.g_base_dims * (2**2),
                name=name + '.%d' % (i + 9))
        deconv = res_block
        for i in range(2):
            rate = 2**(1 - i)
            deconv = deconv2d(
                input=deconv,
                num_filters=cfg.g_base_dims * rate,
                filter_size=4,
                stride=2,
                padding=1,
                use_bias=False,
                norm="instance_norm",
                activation_fn='relu',
                name=name + str(15 + i * 3),
                initial='kaiming')
        out = conv2d(
            input=deconv,
            num_filters=3,
            filter_size=7,
            stride=1,
            padding=3,
            use_bias=False,
            norm=None,
            activation_fn='tanh',
            name=name + '21',
            initial='kaiming')
        return out

    def network_D(self, input, cfg, name="discriminator"):
        conv0 = conv2d(
            input=input,
            num_filters=cfg.d_base_dims,
            filter_size=4,
            stride=2,
            padding=1,
            activation_fn='leaky_relu',
            name=name + '0',
            initial='kaiming')
        repeat_num = 6
        curr_dim = cfg.d_base_dims
        conv = conv0
        for i in range(1, repeat_num):
            curr_dim *= 2
            conv = conv2d(
                input=conv,
                num_filters=curr_dim,
                filter_size=4,
                stride=2,
                padding=1,
                activation_fn='leaky_relu',
                name=name + str(i * 2),
                initial='kaiming')
        kernel_size = int(cfg.image_size / np.power(2, repeat_num))
        out1 = conv2d(
            input=conv,
            num_filters=1,
            filter_size=3,
            stride=1,
            padding=1,
            use_bias=False,
            name="d_conv1",
            initial='kaiming')
        out2 = conv2d(
            input=conv,
            num_filters=cfg.c_dim,
            filter_size=kernel_size,
            use_bias=False,
            name="d_conv2",
            initial='kaiming')
        return out1, out2
