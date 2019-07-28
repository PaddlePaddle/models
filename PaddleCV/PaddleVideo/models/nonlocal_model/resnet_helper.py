#  Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
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

from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import paddle
import paddle.fluid as fluid
from paddle.fluid import ParamAttr

import numpy as np
from . import nonlocal_helper


def Conv3dAffine(blob_in,
                 prefix,
                 dim_in,
                 dim_out,
                 filter_size,
                 stride,
                 padding,
                 cfg,
                 group=1,
                 test_mode=False,
                 bn_init=None):
    blob_out = fluid.layers.conv3d(
        input=blob_in,
        num_filters=dim_out,
        filter_size=filter_size,
        stride=stride,
        padding=padding,
        groups=group,
        param_attr=ParamAttr(
            name=prefix + "_weights", initializer=fluid.initializer.MSRA()),
        bias_attr=False,
        name=prefix + "_conv")
    blob_out_shape = blob_out.shape

    affine_name = "bn" + prefix[3:]

    affine_scale = fluid.layers.create_parameter(
        shape=[blob_out_shape[1]],
        dtype=blob_out.dtype,
        attr=ParamAttr(name=affine_name + '_scale'),
        default_initializer=fluid.initializer.Constant(value=1.))
    affine_bias = fluid.layers.create_parameter(
        shape=[blob_out_shape[1]],
        dtype=blob_out.dtype,
        attr=ParamAttr(name=affine_name + '_offset'),
        default_initializer=fluid.initializer.Constant(value=0.))
    blob_out = fluid.layers.affine_channel(
        blob_out, scale=affine_scale, bias=affine_bias, name=affine_name)

    return blob_out


def Conv3dBN(blob_in,
             prefix,
             dim_in,
             dim_out,
             filter_size,
             stride,
             padding,
             cfg,
             group=1,
             test_mode=False,
             bn_init=None):
    blob_out = fluid.layers.conv3d(
        input=blob_in,
        num_filters=dim_out,
        filter_size=filter_size,
        stride=stride,
        padding=padding,
        groups=group,
        param_attr=ParamAttr(
            name=prefix + "_weights", initializer=fluid.initializer.MSRA()),
        bias_attr=False,
        name=prefix + "_conv")

    bn_name = "bn" + prefix[3:]

    blob_out = fluid.layers.batch_norm(
        blob_out,
        is_test=test_mode,
        momentum=cfg.MODEL.bn_momentum,
        epsilon=cfg.MODEL.bn_epsilon,
        name=bn_name,
        param_attr=ParamAttr(
            name=bn_name + "_scale",
            initializer=fluid.initializer.Constant(value=bn_init if
                                                   (bn_init != None) else 1.),
            regularizer=fluid.regularizer.L2Decay(cfg.TRAIN.weight_decay_bn)),
        bias_attr=ParamAttr(
            name=bn_name + "_offset",
            regularizer=fluid.regularizer.L2Decay(cfg.TRAIN.weight_decay_bn)),
        moving_mean_name=bn_name + "_mean",
        moving_variance_name=bn_name + "_variance")
    return blob_out


# 3d bottleneck
def bottleneck_transformation_3d(blob_in,
                                 dim_in,
                                 dim_out,
                                 stride,
                                 prefix,
                                 dim_inner,
                                 cfg,
                                 group=1,
                                 use_temp_conv=1,
                                 temp_stride=1,
                                 test_mode=False):
    conv_op = Conv3dAffine if cfg.MODEL.use_affine else Conv3dBN

    # 1x1 layer
    blob_out = conv_op(
        blob_in,
        prefix + "_branch2a",
        dim_in,
        dim_inner, [1 + use_temp_conv * 2, 1, 1], [temp_stride, 1, 1],
        [use_temp_conv, 0, 0],
        cfg,
        test_mode=test_mode)
    blob_out = fluid.layers.relu(blob_out, name=prefix + "_branch2a" + "_relu")

    # 3x3 layer
    blob_out = conv_op(
        blob_out,
        prefix + '_branch2b',
        dim_inner,
        dim_inner, [1, 3, 3], [1, stride, stride], [0, 1, 1],
        cfg,
        group=group,
        test_mode=test_mode)
    blob_out = fluid.layers.relu(blob_out, name=prefix + "_branch2b" + "_relu")

    # 1x1 layer, no relu
    blob_out = conv_op(
        blob_out,
        prefix + '_branch2c',
        dim_inner,
        dim_out, [1, 1, 1], [1, 1, 1], [0, 0, 0],
        cfg,
        test_mode=test_mode,
        bn_init=cfg.MODEL.bn_init_gamma)

    return blob_out


def _add_shortcut_3d(blob_in,
                     prefix,
                     dim_in,
                     dim_out,
                     stride,
                     cfg,
                     temp_stride=1,
                     test_mode=False):
    if ((dim_in == dim_out) and (temp_stride == 1) and (stride == 1)):
        # identity mapping (do nothing)
        return blob_in
    else:
        # when dim changes
        conv_op = Conv3dAffine if cfg.MODEL.use_affine else Conv3dBN
        blob_out = conv_op(
            blob_in,
            prefix,
            dim_in,
            dim_out, [1, 1, 1], [temp_stride, stride, stride], [0, 0, 0],
            cfg,
            test_mode=test_mode)

        return blob_out


# residual block abstraction
def _generic_residual_block_3d(blob_in,
                               dim_in,
                               dim_out,
                               stride,
                               prefix,
                               dim_inner,
                               cfg,
                               group=1,
                               use_temp_conv=0,
                               temp_stride=1,
                               trans_func=None,
                               test_mode=False):
    # transformation branch (e.g. 1x1-3x3-1x1, or 3x3-3x3), namely "F(x)"
    if trans_func is None:
        trans_func = globals()[cfg.RESNETS.trans_func]

    tr_blob = trans_func(
        blob_in,
        dim_in,
        dim_out,
        stride,
        prefix,
        dim_inner,
        cfg,
        group=group,
        use_temp_conv=use_temp_conv,
        temp_stride=temp_stride,
        test_mode=test_mode)

    # create short cut, namely, "x"
    sc_blob = _add_shortcut_3d(
        blob_in,
        prefix + "_branch1",
        dim_in,
        dim_out,
        stride,
        cfg,
        temp_stride=temp_stride,
        test_mode=test_mode)

    # addition, namely, "x + F(x)", and relu
    sum_blob = fluid.layers.elementwise_add(
        tr_blob, sc_blob, act='relu', name=prefix + '_sum')

    return sum_blob


def res_stage_nonlocal(block_fn,
                       blob_in,
                       dim_in,
                       dim_out,
                       stride,
                       num_blocks,
                       prefix,
                       cfg,
                       dim_inner=None,
                       group=None,
                       use_temp_convs=None,
                       temp_strides=None,
                       batch_size=None,
                       nonlocal_name=None,
                       nonlocal_mod=1000,
                       test_mode=False):
    # prefix is something like: res2, res3, etc.
    # each res layer has num_blocks stacked.

    # check dtype and format of use_temp_convs and temp_strides
    if use_temp_convs is None:
        use_temp_convs = np.zeros(num_blocks).astype(int)
    if temp_strides is None:
        temp_strides = np.ones(num_blocks).astype(int)

    if len(use_temp_convs) < num_blocks:
        for _ in range(num_blocks - len(use_temp_convs)):
            use_temp_convs.append(0)
            temp_strides.append(1)

    for idx in range(num_blocks):
        block_prefix = '{}{}'.format(prefix, chr(idx + 97))
        if cfg.MODEL.depth == 101:
            if num_blocks == 23:
                if idx == 0:
                    block_prefix = '{}{}'.format(prefix, chr(97))
                else:
                    block_prefix = '{}{}{}'.format(prefix, 'b', idx)

        block_stride = 2 if ((idx == 0) and (stride == 2)) else 1
        blob_in = _generic_residual_block_3d(
            blob_in,
            dim_in,
            dim_out,
            block_stride,
            block_prefix,
            dim_inner,
            cfg,
            group=group,
            use_temp_conv=use_temp_convs[idx],
            temp_stride=temp_strides[idx],
            test_mode=test_mode)
        dim_in = dim_out

        if idx % nonlocal_mod == nonlocal_mod - 1:
            blob_in = nonlocal_helper.add_nonlocal(
                blob_in,
                dim_in,
                dim_in,
                batch_size,
                nonlocal_name + '_{}'.format(idx),
                int(dim_in / 2),
                cfg,
                test_mode=test_mode)

    return blob_in, dim_in


def res_stage_nonlocal_group(block_fn,
                             blob_in,
                             dim_in,
                             dim_out,
                             stride,
                             num_blocks,
                             prefix,
                             cfg,
                             dim_inner=None,
                             group=None,
                             use_temp_convs=None,
                             temp_strides=None,
                             batch_size=None,
                             pool_stride=None,
                             spatial_dim=None,
                             group_size=None,
                             nonlocal_name=None,
                             nonlocal_mod=1000,
                             test_mode=False):
    # prefix is something like res2, res3, etc.
    # each res layer has num_blocks stacked

    # check dtype and format of use_temp_convs and temp_strides
    if use_temp_convs is None:
        use_temp_convs = np.zeros(num_blocks).astype(int)
    if temp_strides is None:
        temp_strides = np.ones(num_blocks).astype(int)

    for idx in range(num_blocks):
        block_prefix = "{}{}".format(prefix, chr(idx + 97))
        block_stride = 2 if (idx == 0 and stride == 2) else 1
        blob_in = _generic_residual_block_3d(
            blob_in,
            dim_in,
            dim_out,
            block_stride,
            block_prefix,
            dim_inner,
            cfg,
            group=group,
            use_temp_conv=use_temp_convs[idx],
            temp_stride=temp_strides[idx],
            test_mode=test_mode)
        dim_in = dim_out

        if idx % nonlocal_mod == nonlocal_mod - 1:
            blob_in = nonlocal_helper.add_nonlocal_group(
                blob_in,
                dim_in,
                dim_in,
                batch_size,
                pool_stride,
                spatial_dim,
                spatial_dim,
                group_size,
                nonlocal_name + "_{}".format(idx),
                int(dim_in / 2),
                cfg,
                test_mode=test_mode)

    return blob_in, dim_in
