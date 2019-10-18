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

from .base_network import linear, conv2d, deconv2d, conv_cond_concat

import paddle.fluid as fluid
import numpy as np
import time
import os
import sys


class CGAN_model(object):
    def __init__(self, batch_size=1):
        self.batch_size = batch_size
        self.img_w = 28
        self.img_h = 28

        self.y_dim = 1
        self.gf_dim = 128
        self.df_dim = 64
        self.leaky_relu_factor = 0.2
        if self.batch_size == 1:
            self.norm = None
        else:
            self.norm = "batch_norm"

    def network_G(self, input, label, name="generator"):
        # concat noise and label
        y = fluid.layers.reshape(label, shape=[-1, self.y_dim, 1, 1])
        xy = fluid.layers.concat([input, y], 1)
        o_l1 = linear(
            input=xy,
            output_size=self.gf_dim * 8,
            norm=self.norm,
            activation_fn='relu',
            name=name + '_l1')
        o_c1 = fluid.layers.concat([o_l1, y], 1)
        o_l2 = linear(
            input=o_c1,
            output_size=self.gf_dim * (self.img_w // 4) * (self.img_h // 4),
            norm=self.norm,
            activation_fn='relu',
            name=name + '_l2')
        o_r1 = fluid.layers.reshape(
            o_l2,
            shape=[-1, self.gf_dim, self.img_w // 4, self.img_h // 4],
            name=name + '_reshape')
        o_c2 = conv_cond_concat(o_r1, y)
        o_dc1 = deconv2d(
            input=o_c2,
            num_filters=self.gf_dim,
            filter_size=4,
            stride=2,
            padding=[1, 1],
            norm='batch_norm',
            activation_fn='relu',
            name=name + '_dc1',
            output_size=[self.img_w // 2, self.img_h // 2])
        o_c3 = conv_cond_concat(o_dc1, y)
        o_dc2 = deconv2d(
            input=o_dc1,
            num_filters=1,
            filter_size=4,
            stride=2,
            padding=[1, 1],
            activation_fn='tanh',
            name=name + '_dc2',
            output_size=[self.img_w, self.img_h])
        out = fluid.layers.reshape(o_dc2, [-1, self.img_w * self.img_h])
        return out

    def network_D(self, input, label, name="discriminator"):
        # concat image and label
        x = fluid.layers.reshape(input, shape=[-1, 1, self.img_w, self.img_h])
        y = fluid.layers.reshape(label, shape=[-1, self.y_dim, 1, 1])
        xy = conv_cond_concat(x, y)
        o_l1 = conv2d(
            input=xy,
            num_filters=self.df_dim,
            filter_size=3,
            stride=2,
            name=name + '_l1',
            activation_fn='leaky_relu')
        o_c1 = conv_cond_concat(o_l1, y)
        o_l2 = conv2d(
            input=o_c1,
            num_filters=self.df_dim,
            filter_size=3,
            stride=2,
            name=name + '_l2',
            norm='batch_norm',
            activation_fn='leaky_relu')
        o_f1 = fluid.layers.flatten(o_l2, axis=1)
        o_c2 = fluid.layers.concat([o_f1, y], 1)
        o_l3 = linear(
            input=o_c2,
            output_size=self.df_dim * 16,
            norm=self.norm,
            activation_fn='leaky_relu',
            name=name + '_l3')
        o_c3 = fluid.layers.concat([o_l3, y], 1)
        o_logit = linear(o_c3, 1, activation_fn='sigmoid', name=name + '_l4')
        return o_logit
