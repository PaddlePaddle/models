from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .base_network import conv2d, deconv2d, linear

import paddle.fluid as fluid
import numpy as np
import os


class DCGAN_model(object):
    def __init__(self, cfg, batch_size=1):
        self.batch_size = batch_size
        self.img_dim = 28
        self.gfc_dim = 2048
        self.dfc_dim = 1024
        self.gf_dim = 64
        self.df_dim = 64

    def network_G(self, input, name="generator"):
        o_l1 = linear(input, self.gfc_dim, norm='batch_norm', name=name + '_l1')
        o_l2 = linear(
            o_l1,
            self.gf_dim * 2 * self.img_dim // 4 * self.img_dim // 4,
            norm='batch_norm',
            name=name + '_l2')
        o_r1 = fluid.layers.reshape(
            o_l2, [-1, self.df_dim * 2, self.img_dim // 4, self.img_dim // 4])
        o_dc1 = deconv2d(
            o_r1,
            self.gf_dim * 2,
            4,
            2,
            padding=[1, 1],
            activation_fn='relu',
            output_size=[self.img_dim // 2, self.img_dim // 2],
            name=name + '_dc1')
        o_dc2 = deconv2d(
            o_dc1,
            1,
            4,
            2,
            padding=[1, 1],
            activation_fn='tanh',
            output_size=[self.img_dim, self.img_dim],
            name=name + '_dc2')
        out = fluid.layers.reshape(o_dc2, shape=[-1, 28 * 28])
        return out

    def network_D(self, input, name="discriminator"):
        o_r1 = fluid.layers.reshape(
            input, shape=[-1, 1, self.img_dim, self.img_dim])
        o_c1 = conv2d(
            o_r1,
            self.df_dim,
            4,
            2,
            padding=[1, 1],
            activation_fn='leaky_relu',
            name=name + '_c1')
        o_c2 = conv2d(
            o_c1,
            self.df_dim * 2,
            4,
            2,
            padding=[1, 1],
            norm='batch_norm',
            activation_fn='leaky_relu',
            name=name + '_c2')
        o_l1 = linear(
            o_c2,
            self.dfc_dim,
            norm='batch_norm',
            activation_fn='leaky_relu',
            name=name + '_l1')
        out = linear(o_l1, 1, activation_fn='sigmoid', name=name + '_l2')
        return out
