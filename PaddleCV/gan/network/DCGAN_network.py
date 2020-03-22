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

from .base_network import norm_layer, deconv2d, linear, conv_and_pool

import paddle.fluid as fluid
import numpy as np
import os


class DCGAN_model(object):
    def __init__(self, batch_size=1):
        self.batch_size = batch_size
        self.img_dim = 28
        self.gfc_dim = 2048
        self.dfc_dim = 1024
        self.gf_dim = 64
        self.df_dim = 64
        if self.batch_size == 1:
            self.norm = None
        else:
            self.norm = "batch_norm"

    def network_G(self, input, name="generator"):
        o_l1 = linear(
            input,
            self.gfc_dim,
            norm=self.norm,
            activation_fn='relu',
            name=name + '_l1')
        o_l2 = linear(
            input=o_l1,
            output_size=self.gf_dim * 2 * self.img_dim // 4 * self.img_dim // 4,
            norm=self.norm,
            activation_fn='relu',
            name=name + '_l2')
        o_r1 = fluid.layers.reshape(
            o_l2, [-1, self.df_dim * 2, self.img_dim // 4, self.img_dim // 4])
        o_dc1 = deconv2d(
            input=o_r1,
            num_filters=self.gf_dim * 2,
            filter_size=5,
            stride=2,
            padding=2,
            activation_fn='relu',
            output_size=[self.img_dim // 2, self.img_dim // 2],
            use_bias=True,
            name=name + '_dc1')
        o_dc2 = deconv2d(
            input=o_dc1,
            num_filters=1,
            filter_size=5,
            stride=2,
            padding=2,
            activation_fn='tanh',
            use_bias=True,
            output_size=[self.img_dim, self.img_dim],
            name=name + '_dc2')
        out = fluid.layers.reshape(o_dc2, shape=[-1, 28 * 28])
        return out

    def network_D(self, input, name="discriminator"):
        o_r1 = fluid.layers.reshape(
            input, shape=[-1, 1, self.img_dim, self.img_dim])
        o_c1 = conv_and_pool(
            o_r1, self.df_dim, name=name + '_c1', act='leaky_relu')
        o_c2_1 = conv_and_pool(o_c1, self.df_dim * 2, name=name + '_c2')
        o_c2_2 = norm_layer(
            o_c2_1, norm_type='batch_norm', name=name + '_c2_bn')
        o_c2 = fluid.layers.leaky_relu(o_c2_2, name=name + '_c2_leaky_relu')
        o_l1 = linear(
            input=o_c2,
            output_size=self.dfc_dim,
            norm=self.norm,
            activation_fn='leaky_relu',
            name=name + '_l1')
        out = linear(o_l1, 1, activation_fn=None, name=name + '_l2')
        return out
