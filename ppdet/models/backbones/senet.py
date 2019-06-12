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
import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.framework import Variable
from paddle.fluid.regularizer import L2Decay

from ..registry import Backbones
from ..registry import BBoxHeadConvs
from .base import BackboneBase
from .resnet_vd import ResNetVd

__all__ = ['SENet154Backbone', 'SENet154C5']


class SENet(ResNetVd):
    def __init__(self, depth, freeze_bn, affine_channel, groups, bn_decay=True):
        """
        Args:
            depth (int): SENet depth, should be 50, 101, 152.
            freeze_bn (bool): whether to fix batch norm
                (meaning the scale and bias does not update).
            affine_channel (bool): Use batch_norm or affine_channel.
            groups (int): define group parammeter for bottleneck
            bn_decay (bool): Wether perform L2Decay in batch_norm offset
                             and scale, default True.
        """
        if depth not in [50, 101, 152]:
            raise ValueError("depth {} not in [50, 101, 152].".format(depth))
        self.depth = depth
        self.freeze_bn = freeze_bn
        self.affine_channel = affine_channel
        self.bn_decay = bn_decay
        self.depth_cfg = {
            50: ([3, 4, 6, 3], self.bottleneck),
            101: ([3, 4, 23, 3], self.bottleneck),
            152: ([3, 8, 36, 3], self.bottleneck)
        }
        if depth < 152:
            self.stage_filters = [128, 256, 512, 1024]
        else:
            self.stage_filters = [256, 512, 1024, 2048]
        self.groups = groups
        self.reduction_ratio = 16

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
            bias_attr=False)
        bn_name = name + "_bn"
        lr = 0. if self.freeze_bn else 1.
        bn_decay = float(self.bn_decay)
        pattr = ParamAttr(
            name=bn_name + '_scale',
            learning_rate=lr,
            regularizer=L2Decay(bn_decay))
        battr = ParamAttr(
            name=bn_name + '_offset',
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

    def _conv_norm_vd(self,
                      input,
                      num_filters,
                      filter_size,
                      stride=1,
                      groups=1,
                      act=None,
                      name=None):
        pool = fluid.layers.pool2d(
            input=input,
            pool_size=2,
            pool_stride=2,
            pool_padding=0,
            ceil_mode=True,
            pool_type='avg')
        conv = fluid.layers.conv2d(
            input=pool,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=1,
            padding=(filter_size - 1) // 2,
            groups=groups,
            act=None,
            param_attr=ParamAttr(name=name + "_weights"),
            bias_attr=False)
        bn_name = name + "_bn"
        lr = 0. if self.freeze_bn else 1.
        bn_decay = float(self.bn_decay)
        pattr = ParamAttr(
            name=bn_name + '_scale',
            learning_rate=lr,
            regularizer=L2Decay(bn_decay))
        battr = ParamAttr(
            name=bn_name + '_offset',
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

    def _squeeze_excitation(self,
                            input,
                            num_channels,
                            reduction_ratio,
                            name=None):
        pool = fluid.layers.pool2d(
            input=input,
            pool_size=0,
            pool_type='avg',
            global_pooling=True,
            use_cudnn=False)
        stdv = 1.0 / math.sqrt(pool.shape[1] * 1.0)
        squeeze = fluid.layers.fc(
            input=pool,
            size=int(num_channels / reduction_ratio),
            act='relu',
            param_attr=fluid.param_attr.ParamAttr(
                initializer=fluid.initializer.Uniform(-stdv, stdv),
                name=name + '_sqz_weights'),
            bias_attr=ParamAttr(name=name + '_sqz_offset'))
        stdv = 1.0 / math.sqrt(squeeze.shape[1] * 1.0)
        excitation = fluid.layers.fc(
            input=squeeze,
            size=num_channels,
            act='sigmoid',
            param_attr=fluid.param_attr.ParamAttr(
                initializer=fluid.initializer.Uniform(-stdv, stdv),
                name=name + '_exc_weights'),
            bias_attr=ParamAttr(name=name + '_exc_offset'))
        scale = fluid.layers.elementwise_mul(x=input, y=excitation, axis=0)
        return scale

    def bottleneck(self,
                   input,
                   num_filters,
                   stride,
                   groups,
                   reduction_ratio,
                   is_first,
                   name=None):
        conv0 = self._conv_norm(
            input=input,
            num_filters=num_filters,
            filter_size=1,
            act='relu',
            name='conv' + name + '_x1')
        conv1 = self._conv_norm(
            input=conv0,
            num_filters=num_filters,
            filter_size=3,
            stride=stride,
            groups=groups,
            act='relu',
            name='conv' + name + '_x2')
        if groups == 64:
            num_filters = int(num_filters / 2)
        conv2 = self._conv_norm(
            input=conv1,
            num_filters=num_filters * 2,
            filter_size=1,
            act=None,
            name='conv' + name + '_x3')
        scale = self._squeeze_excitation(
            input=conv2,
            num_channels=num_filters * 2,
            reduction_ratio=reduction_ratio,
            name='fc' + name)

        short = self._shortcut(
            input, num_filters * 2, stride, is_first=is_first, name=name)

        return fluid.layers.elementwise_add(x=short, y=scale, act='relu')

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
        ch_out = self.stage_filters[stage_num - 2]
        is_first = False if stage_num != 2 else True
        conv = input
        n = stage_num + 2
        for i in range(count):
            conv = block_func(
                input=conv,
                num_filters=ch_out,
                stride=2 if i == 0 and stage_num != 2 else 1,
                groups=self.groups,
                reduction_ratio=self.reduction_ratio,
                is_first=is_first,
                name=str(n) + '_' + str(i + 1))
        return conv

    def c1_stage(self, input):
        conv = self._conv_norm(
            input=input,
            num_filters=64,
            filter_size=3,
            stride=2,
            act='relu',
            name='conv1_1')
        conv = self._conv_norm(
            input=conv,
            num_filters=64,
            filter_size=3,
            stride=1,
            act='relu',
            name='conv1_2')
        conv = self._conv_norm(
            input=conv,
            num_filters=128,
            filter_size=3,
            stride=1,
            act='relu',
            name='conv1_3')
        conv = fluid.layers.pool2d(
            input=conv,
            pool_size=3,
            pool_stride=2,
            pool_padding=1,
            pool_type='max')
        return conv


@Backbones.register
class SENet154Backbone(BackboneBase):
    def __init__(self, cfg):
        super(SENet154Backbone, self).__init__(cfg)
        self.freeze_at = getattr(cfg.MODEL, 'FREEZE_AT', 2)
        assert self.freeze_at in [0, 2, 3, 4
                                  ], "The freeze_at should be 0, 2, 3, or 4"
        self.freeze_bn = getattr(cfg.MODEL, 'FREEZE_BN', False)
        self.affine_channel = getattr(cfg.MODEL, 'AFFINE_CHANNEL', False)
        # whether ignore batch_norm offset and scale L2Decay
        self.bn_decay = getattr(cfg.OPTIMIZER.WEIGHT_DECAY, 'BN_DECAY', True)
        self.groups = getattr(cfg.MODEL, "GROUPS", 64)
        self.endpoint = getattr(cfg.MODEL, 'ENDPOINT', 4)
        self.number = 152
        # This list contains names of each Res Block output.
        # The name is the key of body_dict as well. 
        self.body_feat_names = [
            'res' + str(lvl) + '_sum' for lvl in range(2, self.endpoint + 1)
        ]

    def __call__(self, input):
        if not isinstance(input, Variable):
            raise TypeError(str(input) + " shouble be Variable")
        model = SENet(self.number, self.freeze_bn, self.affine_channel,
                      self.groups, self.bn_decay)
        res_list = model.get_backbone(input, self.endpoint, self.freeze_at)
        return {k: v for k, v in zip(self.body_feat_names, res_list)}

    # TODO(guanzhong): add more comments.
    def get_body_feat_names(self):
        return self.body_feat_names


@BBoxHeadConvs.register
class SENet154C5(object):
    def __init__(self, cfg):
        self.freeze_bn = getattr(cfg.MODEL, 'FREEZE_BN', False)
        self.affine_channel = getattr(cfg.MODEL, 'AFFINE_CHANNEL', False)
        self.groups = getattr(cfg.MODEL, 'GROUPS', 64)
        self.number = 152
        self.stage_number = 5

    def __call__(self, input):
        model = SENet(self.number, self.freeze_bn, self.affine_channel,
                      self.groups)
        res5 = model.layer_warp(input, self.stage_number)
        feat = fluid.layers.pool2d(
            input=res5,
            pool_size=7,
            pool_type='avg',
            global_pooling=True,
            use_cudnn=False)
        feat = fluid.layers.dropout(x=feat, dropout_prob=0.2)
        return feat
