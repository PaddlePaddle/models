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

import math

import paddle.fluid as fluid
from paddle.fluid.initializer import MSRA
from paddle.fluid.param_attr import ParamAttr

__all__ = [
    'ShuffleNetV2_x0_5_swish', 'ShuffleNetV2_x1_0_swish',
    'ShuffleNetV2_x1_5_swish', 'ShuffleNetV2_x2_0_swish', 'ShuffleNetV2_swish'
]


class ShuffleNetV2_swish():
    def __init__(self, scale=1.0):
        self.scale = scale

    def net(self, input, class_dim=1000):
        scale = self.scale
        stage_repeats = [4, 8, 4]

        if scale == 0.5:
            stage_out_channels = [-1, 24, 48, 96, 192, 1024]
        elif scale == 1.0:
            stage_out_channels = [-1, 24, 116, 232, 464, 1024]
        elif scale == 1.5:
            stage_out_channels = [-1, 24, 176, 352, 704, 1024]
        elif scale == 2.0:
            stage_out_channels = [-1, 24, 224, 488, 976, 2048]
        else:
            raise ValueError("""{} groups is not supported for
                       1x1 Grouped Convolutions""".format(num_groups))

        #conv1

        input_channel = stage_out_channels[1]
        conv1 = self.conv_bn_layer(
            input=input,
            filter_size=3,
            num_filters=input_channel,
            padding=1,
            stride=2,
            name='stage1_conv')
        pool1 = fluid.layers.pool2d(
            input=conv1,
            pool_size=3,
            pool_stride=2,
            pool_padding=1,
            pool_type='max')
        conv = pool1
        # bottleneck sequences
        for idxstage in range(len(stage_repeats)):
            numrepeat = stage_repeats[idxstage]
            output_channel = stage_out_channels[idxstage + 2]
            for i in range(numrepeat):
                if i == 0:
                    conv = self.inverted_residual_unit(
                        input=conv,
                        num_filters=output_channel,
                        stride=2,
                        benchmodel=2,
                        name=str(idxstage + 2) + '_' + str(i + 1))
                else:
                    conv = self.inverted_residual_unit(
                        input=conv,
                        num_filters=output_channel,
                        stride=1,
                        benchmodel=1,
                        name=str(idxstage + 2) + '_' + str(i + 1))

        conv_last = self.conv_bn_layer(
            input=conv,
            filter_size=1,
            num_filters=stage_out_channels[-1],
            padding=0,
            stride=1,
            name='conv5')
        pool_last = fluid.layers.pool2d(
            input=conv_last,
            pool_size=7,
            pool_stride=1,
            pool_padding=0,
            pool_type='avg')

        output = fluid.layers.fc(input=pool_last,
                                 size=class_dim,
                                 param_attr=ParamAttr(
                                     initializer=MSRA(), name='fc6_weights'),
                                 bias_attr=ParamAttr(name='fc6_offset'))
        return output

    def conv_bn_layer(self,
                      input,
                      filter_size,
                      num_filters,
                      stride,
                      padding,
                      num_groups=1,
                      use_cudnn=True,
                      if_act=True,
                      name=None):
        conv = fluid.layers.conv2d(
            input=input,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            groups=num_groups,
            act=None,
            use_cudnn=use_cudnn,
            param_attr=ParamAttr(
                initializer=MSRA(), name=name + '_weights'),
            bias_attr=False)
        out = int((input.shape[2] - 1) / float(stride) + 1)
        bn_name = name + '_bn'
        if if_act:
            return fluid.layers.batch_norm(
                input=conv,
                act='swish',
                param_attr=ParamAttr(name=bn_name + "_scale"),
                bias_attr=ParamAttr(name=bn_name + "_offset"),
                moving_mean_name=bn_name + '_mean',
                moving_variance_name=bn_name + '_variance')
        else:
            return fluid.layers.batch_norm(
                input=conv,
                param_attr=ParamAttr(name=bn_name + "_scale"),
                bias_attr=ParamAttr(name=bn_name + "_offset"),
                moving_mean_name=bn_name + '_mean',
                moving_variance_name=bn_name + '_variance')

    def channel_shuffle(self, x, groups):
        batchsize, num_channels, height, width = x.shape[0], x.shape[
            1], x.shape[2], x.shape[3]
        channels_per_group = num_channels // groups

        # reshape
        x = fluid.layers.reshape(
            x=x, shape=[batchsize, groups, channels_per_group, height, width])

        x = fluid.layers.transpose(x=x, perm=[0, 2, 1, 3, 4])

        # flatten
        x = fluid.layers.reshape(
            x=x, shape=[batchsize, num_channels, height, width])

        return x

    def inverted_residual_unit(self,
                               input,
                               num_filters,
                               stride,
                               benchmodel,
                               name=None):
        assert stride in [1, 2], \
            "supported stride are {} but your stride is {}".format([1,2], stride)

        oup_inc = num_filters // 2
        inp = input.shape[1]

        if benchmodel == 1:
            x1, x2 = fluid.layers.split(
                input,
                num_or_sections=[input.shape[1] // 2, input.shape[1] // 2],
                dim=1)

            conv_pw = self.conv_bn_layer(
                input=x2,
                num_filters=oup_inc,
                filter_size=1,
                stride=1,
                padding=0,
                num_groups=1,
                if_act=True,
                name='stage_' + name + '_conv1')

            conv_dw = self.conv_bn_layer(
                input=conv_pw,
                num_filters=oup_inc,
                filter_size=3,
                stride=stride,
                padding=1,
                num_groups=oup_inc,
                if_act=False,
                use_cudnn=False,
                name='stage_' + name + '_conv2')

            conv_linear = self.conv_bn_layer(
                input=conv_dw,
                num_filters=oup_inc,
                filter_size=1,
                stride=1,
                padding=0,
                num_groups=1,
                if_act=True,
                name='stage_' + name + '_conv3')

            out = fluid.layers.concat([x1, conv_linear], axis=1)

        else:
            #branch1
            conv_dw_1 = self.conv_bn_layer(
                input=input,
                num_filters=inp,
                filter_size=3,
                stride=stride,
                padding=1,
                num_groups=inp,
                if_act=False,
                use_cudnn=False,
                name='stage_' + name + '_conv4')

            conv_linear_1 = self.conv_bn_layer(
                input=conv_dw_1,
                num_filters=oup_inc,
                filter_size=1,
                stride=1,
                padding=0,
                num_groups=1,
                if_act=True,
                name='stage_' + name + '_conv5')

            #branch2
            conv_pw_2 = self.conv_bn_layer(
                input=input,
                num_filters=oup_inc,
                filter_size=1,
                stride=1,
                padding=0,
                num_groups=1,
                if_act=True,
                name='stage_' + name + '_conv1')

            conv_dw_2 = self.conv_bn_layer(
                input=conv_pw_2,
                num_filters=oup_inc,
                filter_size=3,
                stride=stride,
                padding=1,
                num_groups=oup_inc,
                if_act=False,
                use_cudnn=False,
                name='stage_' + name + '_conv2')

            conv_linear_2 = self.conv_bn_layer(
                input=conv_dw_2,
                num_filters=oup_inc,
                filter_size=1,
                stride=1,
                padding=0,
                num_groups=1,
                if_act=True,
                name='stage_' + name + '_conv3')
            out = fluid.layers.concat([conv_linear_1, conv_linear_2], axis=1)

        return self.channel_shuffle(out, 2)


def ShuffleNetV2_x0_5_swish():
    model = ShuffleNetV2_swish(scale=0.5)
    return model


def ShuffleNetV2_x1_0_swish():
    model = ShuffleNetV2_swish(scale=1.0)
    return model


def ShuffleNetV2_x1_5_swish():
    model = ShuffleNetV2_swish(scale=1.5)
    return model


def ShuffleNetV2_x2_0_swish():
    model = ShuffleNetV2_swish(scale=2.0)
    return model
