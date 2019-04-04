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

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import paddle
import paddle.fluid as fluid
from paddle.fluid import ParamAttr


# 3d spacetime nonlocal (v1, spatial downsample)
def spacetime_nonlocal(blob_in, dim_in, dim_out, batch_size, prefix, dim_inner, cfg, \
                       test_mode = False, max_pool_stride = 2):
    #------------
    cur = blob_in
    # we do projection to convert each spacetime location to a feature
    # theta original size
    # e.g.,  (8, 1024, 4, 14, 14) => (8, 1024, 4, 14, 14)
    theta = fluid.layers.conv3d(
        input=cur,
        num_filters=dim_inner,
        filter_size=[1, 1, 1],
        stride=[1, 1, 1],
        padding=[0, 0, 0],
        param_attr=ParamAttr(
            name=prefix + '_theta' + "_w",
            initializer=fluid.initializer.Normal(
                loc=0.0, scale=cfg.NONLOCAL.conv_init_std)),
        bias_attr=ParamAttr(
            name=prefix + '_theta' + "_b",
            initializer=fluid.initializer.Constant(value=0.))
        if (cfg.NONLOCAL.no_bias == 0) else False,
        name=prefix + '_theta')
    theta_shape = theta.shape

    # phi and g: half spatial size
    # e.g., (8, 1024, 4, 14, 14) => (8, 1024, 4, 7, 7)
    if cfg.NONLOCAL.use_maxpool:
        max_pool = fluid.layers.pool3d(
            input=cur,
            pool_size=[1, max_pool_stride, max_pool_stride],
            pool_type='max',
            pool_stride=[1, max_pool_stride, max_pool_stride],
            pool_padding=[0, 0, 0],
            name=prefix + '_pool')
    else:
        max_pool = cur

    phi = fluid.layers.conv3d(
        input=max_pool,
        num_filters=dim_inner,
        filter_size=[1, 1, 1],
        stride=[1, 1, 1],
        padding=[0, 0, 0],
        param_attr=ParamAttr(
            name=prefix + '_phi' + "_w",
            initializer=fluid.initializer.Normal(
                loc=0.0, scale=cfg.NONLOCAL.conv_init_std)),
        bias_attr=ParamAttr(
            name=prefix + '_phi' + "_b",
            initializer=fluid.initializer.Constant(value=0.))
        if (cfg.NONLOCAL.no_bias == 0) else False,
        name=prefix + '_phi')
    phi_shape = phi.shape
    g = fluid.layers.conv3d(
        input=max_pool,
        num_filters=dim_inner,
        filter_size=[1, 1, 1],
        stride=[1, 1, 1],
        padding=[0, 0, 0],
        param_attr=ParamAttr(
            name=prefix + '_g' + "_w",
            initializer=fluid.initializer.Normal(
                loc=0.0, scale=cfg.NONLOCAL.conv_init_std)),
        bias_attr=ParamAttr(
            name=prefix + '_g' + "_b",
            initializer=fluid.initializer.Constant(value=0.))
        if (cfg.NONLOCAL.no_bias == 0) else False,
        name=prefix + '_g')
    g_shape = g.shape

    # we have to use explicit batch size (to support arbitrary spacetime size)
    # e.g. (8, 1024, 4, 14, 14) => (8, 1024, 784)
    theta = fluid.layers.reshape(
        theta, [-1, 0, theta_shape[2] * theta_shape[3] * theta_shape[4]])
    theta = fluid.layers.transpose(theta, [0, 2, 1])
    phi = fluid.layers.reshape(
        phi, [-1, 0, phi_shape[2] * phi_shape[3] * phi_shape[4]])
    theta_phi = fluid.layers.matmul(theta, phi, name=prefix + '_affinity')
    g = fluid.layers.reshape(g, [-1, 0, g_shape[2] * g_shape[3] * g_shape[4]])
    if cfg.NONLOCAL.use_softmax:
        if cfg.NONLOCAL.use_scale is True:
            theta_phi_sc = fluid.layers.scale(theta_phi, scale=dim_inner**-.5)
        else:
            theta_phi_sc = theta_phi
        p = fluid.layers.softmax(
            theta_phi_sc, name=prefix + '_affinity' + '_prob')
    else:
        # not clear about what is doing in xlw's code
        p = None  # not implemented
        raise "Not implemented when not use softmax"

    # note g's axis[2] corresponds to p's axis[2]
    # e.g. g(8, 1024, 784_2) * p(8, 784_1, 784_2) => (8, 1024, 784_1)
    p = fluid.layers.transpose(p, [0, 2, 1])
    t = fluid.layers.matmul(g, p, name=prefix + '_y')

    # reshape back
    # e.g. (8, 1024, 784) => (8, 1024, 4, 14, 14)
    t_shape = t.shape
    # print(t_shape)
    # print(theta_shape)
    t_re = fluid.layers.reshape(t, shape=list(theta_shape))
    blob_out = t_re

    blob_out = fluid.layers.conv3d(
        input=blob_out,
        num_filters=dim_out,
        filter_size=[1, 1, 1],
        stride=[1, 1, 1],
        padding=[0, 0, 0],
        param_attr=ParamAttr(
            name=prefix + '_out' + "_w",
            initializer=fluid.initializer.Constant(value=0.)
            if cfg.NONLOCAL.use_zero_init_conv else fluid.initializer.Normal(
                loc=0.0, scale=cfg.NONLOCAL.conv_init_std)),
        bias_attr=ParamAttr(
            name=prefix + '_out' + "_b",
            initializer=fluid.initializer.Constant(value=0.))
        if (cfg.NONLOCAL.no_bias == 0) else False,
        name=prefix + '_out')
    blob_out_shape = blob_out.shape

    if cfg.NONLOCAL.use_bn is True:
        bn_name = prefix + "_bn"
        blob_out = fluid.layers.batch_norm(
            blob_out,
            is_test=test_mode,
            momentum=cfg.NONLOCAL.bn_momentum,
            epsilon=cfg.NONLOCAL.bn_epsilon,
            name=bn_name,
            param_attr=ParamAttr(
                name=bn_name + "_scale",
                initializer=fluid.initializer.Constant(
                    value=cfg.NONLOCAL.bn_init_gamma),
                regularizer=fluid.regularizer.L2Decay(
                    cfg.TRAIN.weight_decay_bn)),
            bias_attr=ParamAttr(
                name=bn_name + "_offset",
                regularizer=fluid.regularizer.L2Decay(
                    cfg.TRAIN.weight_decay_bn)),
            moving_mean_name=bn_name + "_mean",
            moving_variance_name=bn_name + "_variance")  # add bn

    if cfg.NONLOCAL.use_affine is True:
        affine_scale = fluid.layers.create_parameter(
            shape=[blob_out_shape[1]],
            dtype=blob_out.dtype,
            attr=ParamAttr(name=prefix + '_affine' + '_s'),
            default_initializer=fluid.initializer.Constant(value=1.))
        affine_bias = fluid.layers.create_parameter(
            shape=[blob_out_shape[1]],
            dtype=blob_out.dtype,
            attr=ParamAttr(name=prefix + '_affine' + '_b'),
            default_initializer=fluid.initializer.Constant(value=0.))
        blob_out = fluid.layers.affine_channel(
            blob_out,
            scale=affine_scale,
            bias=affine_bias,
            name=prefix + '_affine')  # add affine

    return blob_out


def add_nonlocal(blob_in,
                 dim_in,
                 dim_out,
                 batch_size,
                 prefix,
                 dim_inner,
                 cfg,
                 test_mode=False):
    blob_out = spacetime_nonlocal(blob_in, \
                dim_in, dim_out, batch_size, prefix, dim_inner, cfg, test_mode = test_mode)
    blob_out = fluid.layers.elementwise_add(
        blob_out, blob_in, name=prefix + '_sum')
    return blob_out


# this is to reduce memory usage if the feature maps are big
# devide the feature maps into groups in the temporal dimension,
# and perform non-local operations inside each group.
def add_nonlocal_group(blob_in,
                       dim_in,
                       dim_out,
                       batch_size,
                       pool_stride,
                       height,
                       width,
                       group_size,
                       prefix,
                       dim_inner,
                       cfg,
                       test_mode=False):
    group_num = int(pool_stride / group_size)
    assert (pool_stride % group_size == 0), \
           'nonlocal block {}: pool_stride({}) should be divided by group size({})'.format(prefix, pool_stride, group_size)

    if group_num > 1:
        blob_in = fluid.layers.transpose(
            blob_in, [0, 2, 1, 3, 4], name=prefix + '_pre_trans1')
        blob_in = fluid.layers.reshape(
            blob_in,
            [batch_size * group_num, group_size, dim_in, height, width],
            name=prefix + '_pre_reshape1')
        blob_in = fluid.layers.transpose(
            blob_in, [0, 2, 1, 3, 4], name=prefix + '_pre_trans2')

    blob_out = spacetime_nonlocal(
        blob_in,
        dim_in,
        dim_out,
        batch_size,
        prefix,
        dim_inner,
        cfg,
        test_mode=test_mode)
    blob_out = fluid.layers.elementwise_add(
        blob_out, blob_in, name=prefix + '_sum')

    if group_num > 1:
        blob_out = fluid.layers.transpose(
            blob_out, [0, 2, 1, 3, 4], name=prefix + '_post_trans1')
        blob_out = fluid.layers.reshape(
            blob_out,
            [batch_size, group_num * group_size, dim_out, height, width],
            name=prefix + '_post_reshape1')
        blob_out = fluid.layers.transpose(
            blob_out, [0, 2, 1, 3, 4], name=prefix + '_post_trans2')

    return blob_out
