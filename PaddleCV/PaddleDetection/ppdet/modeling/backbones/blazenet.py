# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

from paddle import fluid
from paddle.fluid.param_attr import ParamAttr

from ppdet.core.workspace import register

__all__ = ['BlazeNet']


@register
class BlazeNet(object):
    """
    BlazeFace, see https://arxiv.org/abs/1907.05047

    Args:
        blaze_filters (list): number of filter for each blaze block
        double_blaze_filters (list): number of filter for each double_blaze block
    """

    def __init__(self,
                 blaze_filters=[[24, 24],[24, 24],[24, 48, 2],[48, 48],[48, 48]],
                 double_blaze_filters=[[48, 24, 96, 2],[96, 24, 96],[96, 24, 96],
                                       [96, 24, 96, 2],[96, 24, 96],[96, 24, 96]],
                 with_extra_blocks=True):
        super(BlazeNet, self).__init__()

        self.blaze_filters = blaze_filters
        self.double_blaze_filters = double_blaze_filters
        self.with_extra_blocks = with_extra_blocks

    def __call__(self, input):
        conv = input
        # first conv
        conv = self._conv_norm(
            input=conv,
            num_filters=24,
            filter_size=3,
            stride=2,
            padding=1,
            act='relu',
            name="conv1")

        for k, v in enumerate(self.blaze_filters):
            assert len(v) in [2, 3], \
                "blaze_filters {} not in [2, 3]".format(len(v))
            if len(v) == 2:
                conv = self.BlazeBlock(conv, v[0], v[1], name='blaze_{}'.format(k))
            elif len(v) == 3:
                conv = self.BlazeBlock(conv, v[0], v[1], stride=v[2], name='blaze_{}'.format(k))

        layers = []
        for k, v in enumerate(self.double_blaze_filters):
            assert len(v) in [3, 4], \
                "blaze_filters {} not in [3, 4]".format(len(v))
            if len(v) == 3:
                conv = self.BlazeBlock(conv, v[0], v[1],
                                       double_channels=v[2], name='double_blaze_{}'.format(k))
            elif len(v) == 4:
                layers.append(conv)
                conv = self.BlazeBlock(conv, v[0], v[1], double_channels=v[2],
                                       stride=v[3], name='double_blaze_{}'.format(k))
        layers.append(conv)

        if not self.with_extra_blocks:
            return layers[-1]
        print("layers' length is {}".format(len(layers)))
        print("{}:{}".format("output1", layers[-2].shape))
        print("{}:{}".format("output2", layers[-1].shape))
        return layers[-2], layers[-1]

    def BlazeBlock(self,
                   input,
                   in_channels,
                   out_channels,
                   double_channels=None,
                   stride=1,
                   name=None):
        assert stride in [1, 2]
        use_pool = not stride == 1
        use_double_block = double_channels is not None
        act = 'relu' if use_double_block else None
        conv_dw = self._conv_norm(
            input=input,
            filter_size=5,
            num_filters=in_channels,
            stride=stride,
            padding=2,
            num_groups=in_channels,
            use_cudnn=False,
            name=name + "1_dw")

        conv_pw = self._conv_norm(
            input=conv_dw,
            filter_size=1,
            num_filters=out_channels,
            stride=1,
            padding=0,
            act=act,
            name=name + "1_sep")

        if use_double_block:
            conv_dw = self._conv_norm(
                input=conv_pw,
                filter_size=5,
                num_filters=out_channels,
                stride=1,
                padding=2,
                use_cudnn=False,
                name=name + "2_dw")

            conv_pw = self._conv_norm(
                input=conv_dw,
                filter_size=1,
                num_filters=double_channels,
                stride=1,
                padding=0,
                name=name + "2_sep")

        # shortcut
        if use_pool:
            shortcut_channel = double_channels or out_channels
            shortcut_pool = self._pooling_block(input, stride, stride)
            channel_pad = self._conv_norm(
                input=shortcut_pool,
                filter_size=1,
                num_filters=shortcut_channel,
                stride=1,
                padding=0,
                name="shortcut" + name)
            return fluid.layers.elementwise_add(x=channel_pad, y=conv_pw, act='relu')
        return fluid.layers.elementwise_add(x=input, y=conv_pw, act='relu')


    def _conv_norm(self,
                   input,
                   filter_size,
                   num_filters,
                   stride,
                   padding,
                   num_groups=1,
                   act='relu',   # None
                   use_cudnn=True,
                   name=None):
        parameter_attr = ParamAttr(
            learning_rate=0.1,
            initializer=fluid.initializer.MSRA(),
            name=name + "_weights")
        conv = fluid.layers.conv2d(
            input=input,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            groups=num_groups,
            act=None,
            use_cudnn=use_cudnn,
            param_attr=parameter_attr,
            bias_attr=False)
        print("{}:{}".format(name, conv.shape))
        return fluid.layers.batch_norm(input=conv, act=act)


    def _pooling_block(self,
                       conv,
                       pool_size,
                       pool_stride,
                       pool_padding=0,
                       ceil_mode=True):
        pool = fluid.layers.pool2d(
            input=conv,
            pool_size=pool_size,
            pool_type='max',
            pool_stride=pool_stride,
            pool_padding=pool_padding,
            ceil_mode=ceil_mode)
        return pool
