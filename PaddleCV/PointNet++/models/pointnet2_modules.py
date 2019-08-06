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
Contains model utility functions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import six
import numpy as np

import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.initializer import Constant


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
    idx = fluid.layers.query_ball(xyz, new_xyz, radius, nsample)
    grouped_xyz = fluid.layers.group_points(xyz, idx)
    expand_new_xyz = fluid.layers.unsqueeze(new_xyz, axes=[2])
    expand_new_xyz = fluid.layers.expand(expand_new_xyz, [1, 1, grouped_xyz.shape[2], 1])
    grouped_xyz -= expand_new_xyz 

    if features is not None:
        grouped_feaures = fluid.layers.group_points(features, idx)
        return fluid.layers.concat([grouped_xyz, grouped_feaures], axis=-1) \
                if use_xyz else grouped_feaures
    else:
        assert use_xyz, "use_xyz should be True when features is None"
        return grouped_xyz


def group_all(xyz, features=None, use_xyz=True):
    """
    Group all xyz and features when npoint is None
    See query_and_group
    """
    grouped_xyz = fluid.layers.unsqueeze(xyz, axes=[2])
    if features is not None:
        grouped_feaures = fluid.layers.unsqueeze(features, axes=[2])
        return fluid.layers.concat([grouped_xyz, grouped_feaures], axis=1) if use_xyz else grouped_feaures
    else:
        return grouped_xyz


def conv_bn(input, out_channels, bn=True, act='relu', name=None):
    param_attr = ParamAttr(name='{}_conv_weight'.format(name),
                           initializer=fluid.initializer.Constant(1.376))
    bias_attr = ParamAttr(initializer=fluid.initializer.Constant(0.213),
                          name='{}_conv_bias'.format(name)) \
                                  if not bn else None
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
                                      param_attr=ParamAttr(initializer=fluid.initializer.Constant(2.673), name=bn_name + "_scale"),
                                      bias_attr=ParamAttr(initializer=fluid.initializer.Constant(1.467), name=bn_name + "_offset"),
                                      moving_mean_name=bn_name + '_mean',
                                      moving_variance_name=bn_name + '_var')

    return out


def MLP(features, out_channels_list, bn=True, act='relu', name=None):
    out = features
    for i, out_channels in enumerate(out_channels_list):
        out = conv_bn(out, out_channels, bn=bn, act=act, name=name + "_{}".format(i))
    return out
        

def pointnet_sa_module_msg(xyz,
                       npoint=None,
                       radiuss=[],
                       nsamples=[],
                       mlps=[],
                       features=None,
                       bn=True,
                       use_xyz=True,
                       name=None):
    """
    PointNet MSG(Multi-Scale Group) Set Abstraction Module

    Args:
        xyz (Variable): xyz coordiantes features with shape [B, N, 3]
        radiuss ([float32]): list of radius of ball
        nsamples ([int32]): list of maximum number of gather features
        mlps ([[int32]]): list of out_channels_list
        features (Variable): features with shape [B, C, N]
        bn (bool): whether perform batch norm after conv2d
        use_xyz (bool): whether use xyz coordiantes features

    Returns:
        new_xyz (Variable): centriods features with shape [B, npoint, 3]
        out (Variable): features with shape [B, npoint, \sum_i{mlps[i][-1]}]
    """
    assert len(radiuss) == len(nsamples) == len(mlps), \
            "radiuss, nsamples, mlps length should be same"

    farthest_idx = fluid.layers.farthest_point_sampling(xyz, npoint)
    new_xyz = fluid.layers.gather_point(xyz, farthest_idx)

    out = None
    if features is not None:
        outs = []
        for i, (radius, nsample, mlp) in enumerate(zip(radiuss, nsamples, mlps)):
            out = query_and_group(xyz, new_xyz, radius, nsample, features, use_xyz) if npoint is not None else group_all(xyz, features, use_xyz)
            out = fluid.layers.transpose(out, perm=[0, 3, 1, 2])
            out = MLP(out, mlp, bn=bn, name=name + '_mlp{}'.format(i)) # TODO(dengkaipeng): mlp[1:] ?
            out = fluid.layers.pool2d(out, pool_size=[1, out.shape[3]], pool_type='max')
            out = fluid.layers.squeeze(out, axes=[-1])
            outs.append(out)
        out = fluid.layers.concat(outs, axis=1)
        out = fluid.layers.transpose(out, perm=[0, 2, 1])

    return (new_xyz, out)


def pointnet_sa_module_ssg(xyz,
                       npoint=None,
                       radius=None,
                       nsample=None,
                       mlp=[],
                       features=None,
                       bn=True,
                       use_xyz=True,
                       name=None):
    """
    PointNet SSG(Single-Scale Group) Set Abstraction Module
    see pointnet_sa_module_msg
    """
    return pointnet_sa_module_msg(xyz, npoint, [radius], [nsample], [mlp], features, bn, use_xyz, name)


def pointnet_fp_module(unknown, known, unknown_feats, known_feats, mlp, bn=True, name=None):
    """
    PointNet Feature Propagation Module

    Args:
        unknown (Variable): unknown xyz coordiantes features with shape [B, N, 3]
        known (Variable): known xyz coordiantes features with shape [B, M, 3]
        unknown_feats (Variable): unknown features with shape [B, N, C1] to be propagated to
        known_feats (Variable): known features with shape [B, N, C2] to be propagated from
        mlp ([int32]): out_channels_list
        bn (bool): whether perform batch norm after conv2d

    Returns:
        new_features (Variable): new features with shape [B, N, mlp[-1]]
    """
    if known is None:
        interp_feats = fluid.layers.expand()
    else:
        dist, idx = fluid.layers.three_nn(unknown, known, eps=1e-8)
        dist = fluid.layers.sqrt(dist)
        dist_recip = (dist / dist) / dist; # 1.0 / dist
        norm = fluid.layers.reduce_sum(dist_recip, dim=-1, keep_dim=True)
        interp_feats = fluid.layers.three_interp(known_feats, dist_recip / norm, idx)

    new_features = interp_feats if unknown_feats is None else \
                    fluid.layers.concat([interp_feats, unknown_feats], axis=-1)
    new_features = fluid.layers.transpose(new_features, perm=[0, 2, 1])
    new_features = fluid.layers.unsqueeze(new_features, axes=[-1])
    new_features = MLP(new_features, mlp, bn=bn, name=name + '_mlp')
    new_features = fluid.layers.squeeze(new_features, axes=[-1])
    new_features = fluid.layers.transpose(new_features, perm=[0, 2, 1])
    
    return new_features


if __name__ == "__main__":
    xyz = fluid.layers.data(name='xyz', shape=[9, 3], dtype='float32')
    xyz_feats = fluid.layers.data(name='xyz_feats', shape=[12, 18], dtype='float32')
    new_xyz, out = pointnet_sa_module_msg(xyz, 4, [0.8, 1.6], [6, 3], [[3, 6], [6, 9]], xyz_feats, name="test")

    place = fluid.CUDAPlace(0)
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    np.random.seed(2333)
    xyz_np = np.random.random((2, 9, 3)).astype('float32')
    xyz_feats_np = np.random.random((2, 18, 12)).astype('float32')
    xyz_feats_np = xyz_feats_np.transpose((0, 2, 1))
    # print("xyz: ", xyz_np.shape, xyz_np)
    # print("xyz_feats: ", xyz_feats_np.shape, xyz_feats_np)
    ret = exe.run(fetch_list=[new_xyz.name, out.name], feed={'xyz': xyz_np, 'xyz_feats': xyz_feats_np})
    print("new_xyz: ", ret[0].shape, ret[0])
    print("out: ", ret[1].shape, ret[1].transpose((0, 2, 1)))
    # print("ball_query0: ", ret[2].shape, ret[2]) # "query_ball_0.tmp_0"
    # print("gourped_xyz0: ", ret[3].shape, ret[3].transpose((0, 3, 1, 2))) # "group_points_0.tmp_0"
    # ret[3].tofile('grouped_xyz.data')
    # print("grouped_feaures: ", ret[4].shape, ret[4].transpose((0, 3, 1, 2))) # "group_points_0.tmp_0"
    # ret[4].tofile('grouped_feaures.data')
    # print("gourp0: ", ret[2].shape, ret[2]) # "transpose_0.tmp_0"
    # print("gourp1: ", ret[3].shape, ret[3])
    # ret[2].tofile('group0.data')
    # print("mlp0: ", ret[2].shape, ret[2]) # "batch_norm_1.tmp_3"
    # print("mlp1: ", ret[3].shape, ret[3])
    # print("conv00: ", ret[2].shape, ret[2]) # "conv2d_0.tmp_1"
    # print("conv10: ", ret[3].shape, ret[3])
    # print("ball_query0: ", ret[2].shape, ret[2]) # "query_ball_0.tmp_0"
    # print("ball_query1: ", ret[3].shape, ret[3])
    # print("gourped_xyz0: ", ret[2].shape, ret[2].transpose((0, 3, 1, 2))) # "group_points_0.tmp_0"
    # print("gourped_xyz1: ", ret[3].shape, ret[3].transpose((0, 3, 1, 2)))

    # known = fluid.layers.data(name='known', shape=[9, 3], dtype='float32')
    # unknown = fluid.layers.data(name='unknown', shape=[18, 3], dtype='float32')
    # known_feats = fluid.layers.data(name='known_feats', shape=[9, 4], dtype='float32')
    # unknown_feats = fluid.layers.data(name='unknown_feats', shape=[18, 8], dtype='float32')
    # new_features = pointnet_fp_module(unknown, known, unknown_feats, known_feats, [6], name="test")
    #
    # place = fluid.CUDAPlace(0)
    # exe = fluid.Executor(place)
    # exe.run(fluid.default_startup_program())
    #
    # np.random.seed(2333)
    # known_np = np.random.random((2, 9, 3)).astype('float32')
    # unknown_np = np.random.random((2, 18, 3)).astype('float32')
    # known_feats_np = np.random.random((2, 4, 9)).astype('float32')
    # unknown_feats_np = np.random.random((2, 8, 18)).astype('float32')
    # known_feats_np = known_feats_np.transpose((0, 2, 1))
    # unknown_feats_np = unknown_feats_np.transpose((0, 2, 1))
    # ret = exe.run(fetch_list=[new_features.name], feed={'known': known_np, 'unknown': unknown_np, 'known_feats': known_feats_np, 'unknown_feats': unknown_feats_np})
    # print(ret[0].shape, ret[0].transpose((0, 2, 1)))
