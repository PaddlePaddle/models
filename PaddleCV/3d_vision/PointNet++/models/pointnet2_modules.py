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
"""
Contains PointNet++  utility functions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.initializer import Constant
from ext_op import *

__all__ = ["conv_bn", "pointnet_sa_module", "pointnet_fp_module","fc_bn"]


def query_and_group(xyz, new_xyz, radius, nsample, features=None, use_xyz=True):
    """
    Perform query_ball and group_points

    Args:
        xyz (Variable): xyz coordiantes features with shape [B, N, 3]
        new_xyz (Variable): centriods features with shape [B, npoint, 3]
        radius (float32): radius of ball
        nsample (int32): maximum number of gather features
        features (Variable): features with shape [B, N, C]
        use_xyz (bool): whether use xyz coordiantes features

    Returns:
        out (Variable): features with shape [B, npoint, nsample, C + 3]
    """
    idx = query_ball(xyz, new_xyz, radius, nsample)
    idx.stop_gradient = True
    xyz = fluid.layers.transpose(xyz,perm=[0, 2, 1])
    grouped_xyz = group_points(xyz, idx)
    expand_new_xyz = fluid.layers.unsqueeze(fluid.layers.transpose(new_xyz, perm=[0, 2, 1]), axes=[-1])
    expand_new_xyz = fluid.layers.expand(expand_new_xyz, [1, 1, 1, grouped_xyz.shape[3]])
    grouped_xyz -= expand_new_xyz

    if features is not None:
        grouped_features = group_points(features, idx)
        return fluid.layers.concat([grouped_xyz, grouped_features], axis=1) \
                if use_xyz else grouped_features
    else:
        assert use_xyz, "use_xyz should be True when features is None"
        return grouped_xyz


def group_all(xyz, features=None, use_xyz=True):
    """
    Group all xyz and features when npoint is None
    See query_and_group
    """
    xyz = fluid.layers.transpose(xyz,perm=[0,2,1])
    grouped_xyz = fluid.layers.unsqueeze(xyz, axes=[2])
    if features is not None:
        grouped_features = fluid.layers.unsqueeze(features, axes=[2])
        return fluid.layers.concat([grouped_xyz, grouped_features], axis=1) if use_xyz else grouped_features
    else:
        return grouped_xyz


def conv_bn(input, out_channels, bn=True, bn_momentum=0.99, act='relu', name=None):
    param_attr = ParamAttr(name='{}_conv_weight'.format(name),)
    bias_attr = ParamAttr(name='{}_conv_bias'.format(name)) \
                                  if not bn else False
    out = fluid.layers.conv2d(input,
                              num_filters=out_channels,
                              filter_size=1,
                              stride=1,
                              padding=0,
                              dilation=1,
                              param_attr=param_attr,
                              bias_attr=bias_attr,
			      act=act if not bn else None)
    if bn:
        bn_name = name + "_bn"
        out = fluid.layers.batch_norm(out,
                                      act=act,
				      momentum=bn_momentum,
                                      param_attr=ParamAttr(name=bn_name + "_scale"),
                                      bias_attr=ParamAttr(name=bn_name + "_offset"),
                                      moving_mean_name=bn_name + '_mean',
                                      moving_variance_name=bn_name + '_var')

    return out

def fc_bn(input, out_channels, bn=False, bn_momentum=0.99, act='relu', name=None):
    param_attr = ParamAttr(name='{}_fc_weight'.format(name))
    if not bn:
        bias_attr = ParamAttr(name='{}_fc_bias'.format(name))
    else:
        bias_attr = False
    out = fluid.layers.fc(input,
                          size=out_channels,
			  param_attr=param_attr,
			  bias_attr=bias_attr)
    if bn:
        bn_name = name + "_bn"
        out = fluid.layers.batch_norm(out,
                                      momentum=bn_momentum,
                                      param_attr=ParamAttr(name=bn_name + "_scale"),
                                      bias_attr=ParamAttr(name=bn_name + "_offset"),
                                      moving_mean_name=bn_name + '_mean',
                                      moving_variance_name=bn_name + '_var')
    if act == "relu":
        out = fluid.layers.relu(out)
    return out

def MLP(features, out_channels_list, bn=True, bn_momentum=0.99, act='relu', name=None):
    out = features
    for i, out_channels in enumerate(out_channels_list):
        out = conv_bn(out, out_channels, bn=bn, act=act, bn_momentum=bn_momentum, name=name + "_{}".format(i))
    return out
        

def pointnet_sa_module(xyz,
                       npoint=None,
                       radiuss=[],
                       nsamples=[],
                       mlps=[],
                       feature=None,
                       bn=True,
		       bn_momentum=0.99,
                       use_xyz=True,
                       name=None):
    """
    PointNet MSG(Multi-Scale Group) Set Abstraction Module.
    Call with radiuss, nsamples, mlps as single element list for 
    SSG(Single-Scale Group).

    Args:
        xyz (Variable): xyz coordiantes features with shape [B, N, 3]
        radiuss ([float32]): list of radius of ball
        nsamples ([int32]): list of maximum number of gather features
        mlps ([[int32]]): list of out_channels_list
        feature (Variable): features with shape [B, C, N]
        bn (bool): whether perform batch norm after conv2d
	bn_momentum (float): momentum of batch norm
        use_xyz (bool): whether use xyz coordiantes features

    Returns:
        new_xyz (Variable): centriods features with shape [B, npoint, 3]
        out (Variable): features with shape [B, npoint, \sum_i{mlps[i][-1]}]
    """
    assert len(radiuss) == len(nsamples) == len(mlps), \
            "radiuss, nsamples, mlps length should be same"

    farthest_idx = farthest_point_sampling(xyz, npoint)
    farthest_idx.stop_gradient = True
    new_xyz = gather_point(xyz, farthest_idx) if npoint is not None else None

    outs = []
    for i, (radius, nsample, mlp) in enumerate(zip(radiuss, nsamples, mlps)):
        out = query_and_group(xyz, new_xyz, radius, nsample, feature, use_xyz) if npoint is not None else group_all(xyz, feature, use_xyz)
        out = MLP(out, mlp, bn=bn, bn_momentum=bn_momentum, name=name + '_mlp{}'.format(i))
        out = fluid.layers.pool2d(out, pool_size=[1, out.shape[3]], pool_type='max')
        out = fluid.layers.squeeze(out, axes=[-1])
        outs.append(out)
    out = fluid.layers.concat(outs, axis=1)

    return (new_xyz, out)


def pointnet_fp_module(unknown, known, unknown_feats, known_feats, mlp, bn=True, bn_momentum=0.99, name=None):
    """
    PointNet Feature Propagation Module

    Args:
        unknown (Variable): unknown xyz coordiantes features with shape [B, N, 3]
        known (Variable): known xyz coordiantes features with shape [B, M, 3]
        unknown_feats (Variable): unknown features with shape [B, N, C1] to be propagated to
        known_feats (Variable): known features with shape [B, M, C2] to be propagated from
        mlp ([int32]): out_channels_list
        bn (bool): whether perform batch norm after conv2d
        bn_momentum (float): momentum of batch norm

    Returns:
        new_features (Variable): new features with shape [B, N, mlp[-1]]
    """
    if known is None:
        raise NotImplementedError("Not implement known as None currently.")
    else:
        dist, idx = three_nn(unknown, known, eps=0)
        dist.stop_gradient = True
        idx.stop_gradient = True
        dist = fluid.layers.sqrt(dist)
        dist_recip = 1.0 / (dist + 1e-8); # 1.0 / dist
        norm = fluid.layers.reduce_sum(dist_recip, dim=-1, keep_dim=True)
        weight = dist_recip / norm
        weight.stop_gradient = True
        interp_feats = three_interp(known_feats, weight, idx)

    new_features = interp_feats if unknown_feats is None else \
                    fluid.layers.concat([interp_feats, unknown_feats], axis=-1)
    new_features = fluid.layers.transpose(new_features, perm=[0, 2, 1])
    new_features = fluid.layers.unsqueeze(new_features, axes=[-1])
    new_features = MLP(new_features, mlp, bn=bn, bn_momentum=bn_momentum, name=name + '_mlp')
    new_features = fluid.layers.squeeze(new_features, axes=[-1])
    new_features = fluid.layers.transpose(new_features, perm=[0, 2, 1])
    
    return new_features

