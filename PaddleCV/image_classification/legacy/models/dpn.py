#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import numpy as np
import time
import sys
import paddle.fluid as fluid
import math

__all__ = ["DPN", "DPN68", "DPN92", "DPN98", "DPN107", "DPN131"]

train_parameters = {
    "input_size": [3, 224, 224],
    "input_mean": [0.485, 0.456, 0.406],
    "input_std": [0.229, 0.224, 0.225],
    "learning_strategy": {
        "name": "piecewise_decay",
        "batch_size": 256,
        "epochs": [30, 60, 90],
        "steps": [0.1, 0.01, 0.001, 0.0001]
    }
}


class DPN(object):
    def __init__(self, layers=68):
        self.params = train_parameters
        self.layers = layers

    def net(self, input, class_dim=1000):
        # get network args
        args = self.get_net_args(self.layers)
        bws = args['bw']
        inc_sec = args['inc_sec']
        rs = args['bw']
        k_r = args['k_r']
        k_sec = args['k_sec']
        G = args['G']
        init_num_filter = args['init_num_filter']
        init_filter_size = args['init_filter_size']
        init_padding = args['init_padding']

        ## define Dual Path Network

        # conv1
        conv1_x_1 = fluid.layers.conv2d(
            input=input,
            num_filters=init_num_filter,
            filter_size=init_filter_size,
            stride=2,
            padding=init_padding,
            groups=1,
            act=None,
            bias_attr=False)
        conv1_x_1 = fluid.layers.batch_norm(
            input=conv1_x_1, act='relu', is_test=False)
        convX_x_x = fluid.layers.pool2d(
            input=conv1_x_1,
            pool_size=3,
            pool_stride=2,
            pool_padding=1,
            pool_type='max')

        for gc in range(4):
            bw = bws[gc]
            inc = inc_sec[gc]
            R = (k_r * bw) // rs[gc]
            if gc == 0:
                _type1 = 'proj'
                _type2 = 'normal'
            else:
                _type1 = 'down'
                _type2 = 'normal'
            convX_x_x = self.dual_path_factory(convX_x_x, R, R, bw, inc, G,
                                               _type1)
            for i_ly in range(2, k_sec[gc] + 1):
                convX_x_x = self.dual_path_factory(convX_x_x, R, R, bw, inc, G,
                                                   _type2)

        conv5_x_x = fluid.layers.concat(convX_x_x, axis=1)
        conv5_x_x = fluid.layers.batch_norm(
            input=conv5_x_x, act='relu', is_test=False)
        pool5 = fluid.layers.pool2d(
            input=conv5_x_x,
            pool_size=7,
            pool_stride=1,
            pool_padding=0,
            pool_type='avg')

        #stdv = 1.0 / math.sqrt(pool5.shape[1] * 1.0)
        stdv = 0.01
        param_attr = fluid.param_attr.ParamAttr(
            initializer=fluid.initializer.Uniform(-stdv, stdv))
        fc6 = fluid.layers.fc(input=pool5,
                              size=class_dim,
                              param_attr=param_attr)

        return fc6

    def get_net_args(self, layers):
        if layers == 68:
            k_r = 128
            G = 32
            k_sec = [3, 4, 12, 3]
            inc_sec = [16, 32, 32, 64]
            bw = [64, 128, 256, 512]
            r = [64, 64, 64, 64]
            init_num_filter = 10
            init_filter_size = 3
            init_padding = 1
        elif layers == 92:
            k_r = 96
            G = 32
            k_sec = [3, 4, 20, 3]
            inc_sec = [16, 32, 24, 128]
            bw = [256, 512, 1024, 2048]
            r = [256, 256, 256, 256]
            init_num_filter = 64
            init_filter_size = 7
            init_padding = 3
        elif layers == 98:
            k_r = 160
            G = 40
            k_sec = [3, 6, 20, 3]
            inc_sec = [16, 32, 32, 128]
            bw = [256, 512, 1024, 2048]
            r = [256, 256, 256, 256]
            init_num_filter = 96
            init_filter_size = 7
            init_padding = 3
        elif layers == 107:
            k_r = 200
            G = 50
            k_sec = [4, 8, 20, 3]
            inc_sec = [20, 64, 64, 128]
            bw = [256, 512, 1024, 2048]
            r = [256, 256, 256, 256]
            init_num_filter = 128
            init_filter_size = 7
            init_padding = 3
        elif layers == 131:
            k_r = 160
            G = 40
            k_sec = [4, 8, 28, 3]
            inc_sec = [16, 32, 32, 128]
            bw = [256, 512, 1024, 2048]
            r = [256, 256, 256, 256]
            init_num_filter = 128
            init_filter_size = 7
            init_padding = 3
        else:
            raise NotImplementedError
        net_arg = {
            'k_r': k_r,
            'G': G,
            'k_sec': k_sec,
            'inc_sec': inc_sec,
            'bw': bw,
            'r': r
        }
        net_arg['init_num_filter'] = init_num_filter
        net_arg['init_filter_size'] = init_filter_size
        net_arg['init_padding'] = init_padding

        return net_arg

    def dual_path_factory(self,
                          data,
                          num_1x1_a,
                          num_3x3_b,
                          num_1x1_c,
                          inc,
                          G,
                          _type='normal'):
        kw = 3
        kh = 3
        pw = (kw - 1) // 2
        ph = (kh - 1) // 2

        # type
        if _type is 'proj':
            key_stride = 1
            has_proj = True
        if _type is 'down':
            key_stride = 2
            has_proj = True
        if _type is 'normal':
            key_stride = 1
            has_proj = False

        # PROJ
        if type(data) is list:
            data_in = fluid.layers.concat([data[0], data[1]], axis=1)
        else:
            data_in = data

        if has_proj:
            c1x1_w = self.bn_ac_conv(
                data=data_in,
                num_filter=(num_1x1_c + 2 * inc),
                kernel=(1, 1),
                pad=(0, 0),
                stride=(key_stride, key_stride))
            data_o1, data_o2 = fluid.layers.split(
                c1x1_w, num_or_sections=[num_1x1_c, 2 * inc], dim=1)
        else:
            data_o1 = data[0]
            data_o2 = data[1]

        # MAIN
        c1x1_a = self.bn_ac_conv(
            data=data_in, num_filter=num_1x1_a, kernel=(1, 1), pad=(0, 0))
        c3x3_b = self.bn_ac_conv(
            data=c1x1_a,
            num_filter=num_3x3_b,
            kernel=(kw, kh),
            pad=(pw, ph),
            stride=(key_stride, key_stride),
            num_group=G)
        c1x1_c = self.bn_ac_conv(
            data=c3x3_b,
            num_filter=(num_1x1_c + inc),
            kernel=(1, 1),
            pad=(0, 0))

        c1x1_c1, c1x1_c2 = fluid.layers.split(
            c1x1_c, num_or_sections=[num_1x1_c, inc], dim=1)

        # OUTPUTS
        summ = fluid.layers.elementwise_add(x=data_o1, y=c1x1_c1)
        dense = fluid.layers.concat([data_o2, c1x1_c2], axis=1)

        return [summ, dense]

    def bn_ac_conv(self,
                   data,
                   num_filter,
                   kernel,
                   pad,
                   stride=(1, 1),
                   num_group=1):
        bn_ac = fluid.layers.batch_norm(input=data, act='relu', is_test=False)
        bn_ac_conv = fluid.layers.conv2d(
            input=bn_ac,
            num_filters=num_filter,
            filter_size=kernel,
            stride=stride,
            padding=pad,
            groups=num_group,
            act=None,
            bias_attr=False)
        return bn_ac_conv


def DPN68():
    model = DPN(layers=68)
    return model


def DPN92():
    model = DPN(layers=92)
    return model


def DPN98():
    model = DPN(layers=98)
    return model


def DPN107():
    model = DPN(layers=107)
    return model


def DPN131():
    model = DPN(layers=131)
    return model
