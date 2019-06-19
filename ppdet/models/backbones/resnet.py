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

from collections import OrderedDict

from paddle import fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.framework import Variable
from paddle.fluid.regularizer import L2Decay

from ppdet.core.workspace import register, serializable

__all__ = ['ResNet', 'ResNetC5']


@register
@serializable
class ResNet(object):
    """
    Residual Network, see https://arxiv.org/abs/1512.03385
    Args:
        depth (int): ResNet depth, should be 18, 34, 50, 101, 152.
        freeze_at (int): freeze the backbone at which stage
        freeze_bn (bool): fix batch norm weights
        affine_channel (bool): use batch_norm or affine_channel.
        bn_decay (bool): apply weight decay to in batch norm weights
        variant (str): ResNet variant, supports 'a', 'b', 'c', 'd' currently
        feature_maps (list): index of the stages whose feature maps are returned
    """
    def __init__(self,
                 depth=50,
                 freeze_at=2,
                 freeze_bn=True,
                 affine_channel=False,
                 bn_decay=True,
                 variant='a',
                 feature_maps=[2, 3, 4, 5]):
        super(ResNet, self).__init__()

        assert depth in [18, 34, 50, 101, 152], \
            "depth {} not in [18, 34, 50, 101, 152]"
        assert variant in ['a', 'b', 'c', 'd'], "invalid ResNet variant"
        assert 0 <= freeze_at <= 4, "freeze_at should be 0, 1, 2, 3 or 4"
        assert len(feature_maps) > 0, "need one or more feature maps"

        self.depth = depth
        self.freeze_at = freeze_at
        self.freeze_bn = freeze_bn
        self.affine_channel = affine_channel
        self.bn_decay = bn_decay
        self.variant = variant
        self.feature_maps = feature_maps
        self.depth_cfg = {
            18: ([2, 2, 2, 2], self.basicblock),
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
        if getattr(self, '_senet_pretrained_weight_fix', False):
            bn_name = "bn_" + name

        lr = 0. if self.freeze_bn else 1.
        bn_decay = float(self.bn_decay)
        pattr = ParamAttr(name=bn_name + '_scale',
                          learning_rate=lr,
                          regularizer=L2Decay(bn_decay))
        battr = ParamAttr(name=bn_name + '_offset',
                          learning_rate=lr,
                          regularizer=L2Decay(bn_decay))

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
        if self.freeze_bn:
            scale.stop_gradient = True
            bias.stop_gradient = True
        return out

    def _shortcut(self, input, ch_out, stride, is_first, name):
        max_pooling_in_short_cut = self.variant == 'd'
        ch_in = input.shape[1]
        if getattr(self, '_senet_pretrained_weight_fix', False):
            name = 'conv' + name + '_prj'
        if ch_in != ch_out or stride != 1 or is_first:
            if max_pooling_in_short_cut:
                input = fluid.layers.pool2d(
                    input=input,
                    pool_size=2,
                    pool_stride=2,
                    pool_padding=0,
                    ceil_mode=True,
                    pool_type='avg')
                return self._conv_norm(input, ch_out, 1, 1, name=name)
            return self._conv_norm(input, ch_out, 1, stride, name=name)
        else:
            return input

    def bottleneck(self, input, num_filters, stride, is_first, name):
        if self.variant == 'a':
            stride1, stride2 = stride, 1
        else:
            stride1, stride2 = 1, stride

        # ResNeXt
        groups = getattr(self, 'groups', 1)
        group_width = getattr(self, 'group_width', -1)
        if groups == 1:
            expand = 4
        elif (groups * group_width) == 256:
            expand = 1
        else:  # FIXME hard code for now, handles 32x4d, 64x4d and 32x8d
            num_filters = num_filters // 2
            expand = 2

        conv_def = [
            [num_filters, 1, stride1, 'relu', 1, name + "_branch2a"],
            [num_filters, 3, stride2, 'relu', groups, name + "_branch2b"],
            [num_filters * expand, 1, 1, None, 1, name + "_branch2c"]
        ]

        residual = input
        for (c, k, s, act, g, _name) in conv_def:
            print("[01;32m{} c: {}, k: {}, s: {}, g: {}, act: {}[0m".format(
                _name, str(c).ljust(4), k, s, g, act))
            residual = self._conv_norm(
                input=residual,
                num_filters=c,
                filter_size=k,
                stride=s,
                act=act,
                groups=g,
                name=_name)
        short = self._shortcut(
            input,
            num_filters * expand,
            stride,
            is_first=False,
            name=name + "_branch1")
        # Squeeze-and-Excitation
        if callable(getattr(self, '_squeeze_excitation', None)):
            residual = self._squeeze_excitation(
                input=residual,
                num_channels=num_filters * 2,
                name='fc' + name)
        return fluid.layers.elementwise_add(
            x=short, y=residual, act='relu', name=name + ".add.output.5")

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
            stage_num (int): the stage number, should be 2, 3, 4, 5

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
        # FIXME hard code for now
        if getattr(self, '_squeeze_excitation', None) is not None:
            num_out_chan = 128
        else:
            num_out_chan = 64
        if self.variant in ['c', 'd']:
            conv_def = [
                [num_out_chan / 2, 3, 2, "conv1_1"],
                [num_out_chan / 2, 3, 1, "conv1_2"],
                [num_out_chan, 3, 1, "conv1_3"],
            ]
        else:
            conv1_name = "conv1"
            if getattr(self, '_resnext_pretrained_name_fix', False):
                conv1_name = "res_conv1"
            conv_def = [[num_out_chan, 7, 2, conv1_name]]

        for (c, k, s, _name) in conv_def:
            input = self._conv_norm(
                input=input,
                num_filters=c,
                filter_size=k,
                stride=s,
                act='relu',
                name=_name)

        output = fluid.layers.pool2d(
            input=input,
            pool_size=3,
            pool_stride=2,
            pool_padding=1,
            pool_type='max')
        return output

    def __call__(self, input):
        assert isinstance(input, Variable)
        assert not (set(self.feature_maps) - set([2, 3, 4, 5])), \
            "feature maps {} not in [2, 3, 4, 5]".format(self.feature_maps)

        res_endpoints = []

        res = input
        # XXX this may not cover all use case, but enough for now
        if len(self.feature_maps) > 1 and 2 in self.feature_maps:
            res = self.c1_stage(res)
        for i in self.feature_maps:
            res = self.layer_warp(res, i)
            res_endpoints.append(res)
            if self.freeze_at > i:
                res.stop_gradient = True

        if len(res_endpoints) == 1:
            return res_endpoints[0]

        return OrderedDict([('res{}_sum'.format(idx), feat) for idx, feat
                            in enumerate(res_endpoints)])


@register
@serializable
class ResNetC5(ResNet):
    __doc__ = ResNet.__doc__

    def __init__(self,
                 depth=50,
                 freeze_at=2,
                 freeze_bn=True,
                 affine_channel=False,
                 bn_decay=True,
                 variant='a',
                 feature_maps=[5]):
        super(ResNetC5, self).__init__(
            depth, freeze_at, freeze_bn, affine_channel, bn_decay,
            variant, feature_maps)
