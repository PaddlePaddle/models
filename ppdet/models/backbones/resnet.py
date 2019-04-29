#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import math
import six

import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.framework import Variable

__all__ = ['ResNet50Backbone', 'ResNet50C5']


class ResNet(object):
    def __init__(self, depth, fix_bn, affine_channel):
        """
        Args:
            depth (int): ResNet depth, should be 18, 34, 50, 101, 152.
            fix_bn (bool): whether to fix batch norm
                (meaning the scale and bias does not update).
            affine_channel (bool): Use batch_norm or affine_channel.
        """
        if depth not in [18, 34, 50, 101, 152]:
            raise ValueError("depth {} not in [18, 34, 50, 101, 152].".format(
                depth))
        self.depth = depth
        self.fix_bn = fix_bn
        self.affine_channel = affine_channel
        self.depth_cfg = {
            18: ([2, 2, 2, 1], self.basicblock),
            34: ([3, 4, 6, 3], self.basicblock),
            50: ([3, 4, 6, 3], self.bottleneck),
            101: ([3, 4, 23, 3], self.bottleneck),
            152: ([3, 8, 36, 3], self.bottleneck)
        }
        self.stage_filters = [64, 128, 256, 512]

    def _conv_norm(self,
                   input,
                   num_filters,
                   filter_size,
                   stride=1,
                   groups=1,
                   act=None,
                   name=None):
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

        lr = 0. if self.fix_bn else 1.
        pattr = ParamAttr(name=bn_name + '_scale', learning_rate=lr)
        battr = ParamAttr(name=bn_name + '_offset', learning_rate=lr)

        if not self.affine_channel:
            out = fluid.layers.batch_norm(
                input=conv,
                act=act,
                name=bn_name + '.output.1',
                param_attr=pattr,
                bias_attr=battr,
                moving_mean_name=bn_name + '_mean',
                moving_variance_name=bn_name + '_variance', )
            scale = fluid.framework._get_var(pattr.name)
            bias = fluid.framework._get_var(battr.name)
        else:
            scale = fluid.layers.create_parameter(
                shape=[conv.shape[1]],
                dtype=conv.dtype,
                attr=pattr,
                default_initializer=fluid.initializer.Constant(1.))
            bias = fluid.layers.create_parameter(
                shape=[conv.shape[1]],
                dtype=conv.dtype,
                attr=battr,
                default_initializer=fluid.initializer.Constant(0.))
            out = fluid.layers.affine_channel(
                x=conv, scale=scale, bias=bias, act=act)
        if self.fix_bn:
            scale.stop_gradient = True
            bias.stop_gradient = True
        return out

    def _shortcut(self, input, ch_out, stride, is_first, name):
        ch_in = input.shape[1]
        if ch_in != ch_out or stride != 1 or is_first == True:
            return self._conv_norm(input, ch_out, 1, stride, name=name)
        else:
            return input

    def bottleneck(self, input, num_filters, stride, is_first, name):
        conv0 = self._conv_norm(
            input=input,
            num_filters=num_filters,
            filter_size=1,
            act='relu',
            name=name + "_branch2a")
        conv1 = self._conv_norm(
            input=conv0,
            num_filters=num_filters,
            filter_size=3,
            stride=stride,
            act='relu',
            name=name + "_branch2b")
        conv2 = self._conv_norm(
            input=conv1,
            num_filters=num_filters * 4,
            filter_size=1,
            act=None,
            name=name + "_branch2c")
        short = self._shortcut(
            input,
            num_filters * 4,
            stride,
            is_first=False,
            name=name + "_branch1")
        return fluid.layers.elementwise_add(
            x=short, y=conv2, act='relu', name=name + ".add.output.5")

    def basicblock(self, input, num_filters, stride, is_first, name):
        conv0 = self._conv_norm(
            input=input,
            num_filters=num_filters,
            filter_size=3,
            act='relu',
            stride=stride,
            name=name + "_branch2a")
        conv1 = self._conv_norm(
            input=conv0,
            num_filters=num_filters,
            filter_size=3,
            act=None,
            name=name + "_branch2b")
        short = self._shortcut(
            input, num_filters, stride, is_first, name=name + "_branch1")
        return fluid.layers.elementwise_add(x=short, y=conv1, act='relu')

    def layer_warp(self, input, stage_num):
        """
        Args:
            input (Variable): input variable.
            stage_num (int): the stage number, should be 0, 1, 2, 3

        Returns:
            The last variable in endpoint-th stage.
        """
        assert stage_num in [2, 3, 4, 5]

        stages, block_func = self.depth_cfg[self.depth]
        count = stages[stage_num - 2]

        stride = 1 if stage_num == 2 else 2
        ch_out = self.stage_filters[stage_num - 2]
        # Make the layer name and parameter name consistent
        # with ImageNet pre-trained model
        name = 'res' + str(stage_num)

        res_out = block_func(
            input, ch_out, stride, is_first=True, name=name + "a")
        for i in range(1, count):
            conv_name = name + "b" + str(i) if count > 10 else name + chr(
                ord("a") + i)
            res_out = block_func(
                res_out, ch_out, 1, is_first=False, name=conv_name)
        return res_out

    def c1_stage(self, input):
        conv = self._conv_norm(
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
        return conv

    def get_backone(self, input, endpoint, freeze_at=0):
        """
        Get the backbone of ResNet50. We define ResNet50 has 5 stages, from 1 to 5.
        So please carefully to set endpoint and freeze_at number.

        Args:
            input (Variable): input variable.
            endpoint (int): the endpoint stage number, should be 2, 3, 4, or 5.
            freeze_at (int): freeze the backbone at which stage. This number
                should be not large than 4. 0 means that no layers are fixed.

        Returns:
            The last variable in endpoint-th stage.
        """
        if not isinstance(input, Variable):
            raise TypeError(str(input) + " should be Variable")
        if endpoint not in [2, 3, 4, 5]:
            raise ValueError("endpoint {} not in [2, 3, 4, 5].".format(
                endpoint))

        res_endpoints = []

        res = self.c1_stage(input)
        res_endpoints.append(res)
        for i in six.moves.xrange(2, endpoint + 1):
            res = self.layer_warp(res, i)
            res_endpoints.append(res)

        if freeze_at > 0:
            res[freeze_at - 1].stop_gradient = True
        return res


def ResNet50Backbone(input,
                     freeze_at,
                     fix_bn=False,
                     affine_channel=False,
                     bn_type='bn'):
    """
    Get the backbone of ResNet50. We define ResNet50 has 5 stages, from 1 to 5.

    Args:
        freeze_at (int): freeze the backbone at which stage. This number
            should be not large than 4. 0 means that no layers are fixed.
        fix_bn (bool): whether to fix batch norm
            (meaning the scale and bias does not update). Defalut False.
        affine_channel (bool): Use batch_norm or affine_channel.

    Returns:
        The last variable in endpoint-th stage.
    """
    assert freeze_at in [0, 1, 2, 3, 4
                         ], "The freeze_at should be 0, 1, 2, 3 or 4"
    model = ResNet(50, fix_bn, affine_channel)
    return model.get_backone(input, 4, freeze_at)


def ResNet50C5(input, fix_bn=False, affine_channel=False):
    """
    Args:
        fix_bn (bool): whether to fix batch norm
            (meaning the scale and bias does not update). Defalut False.
    Returns:
        The last variable in C5 stage.
    """
    model = ResNet(50, fix_bn, affine_channel)
    out = model.layer_warp(input, 5)
    return out
