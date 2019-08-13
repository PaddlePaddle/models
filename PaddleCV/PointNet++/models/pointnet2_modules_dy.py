#  Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Contains model utility functions
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.initializer import Constant
from paddle.fluid.dygraph.nn import Conv2D, BatchNorm, FC

__all__ = ["squeeze", "unsqueeze", "ConvBN", "Pointnet_SA_Module_MSG", "Pointnet_FP_module", "FCBN"]


def squeeze(var, axis=-1):
    shape = list(var.shape)
    assert shape[axis] == 1
    shape.pop(axis)
    return fluid.layers.reshape(var, shape)

def unsqueeze(var, axis=-1):
    shape = list(var.shape)
    rank = len(shape)
    orig_axis = axis
    if axis == -1:
        new_shape = shape + [1]
    else:
        axis = axis % rank
        new_shape = []
        for i in range(axis):
            new_shape.append(shape[i])
        new_shape.append(1)
        for i in range(axis, rank):
            new_shape.append(shape[i])
    return fluid.layers.reshape(var, new_shape)


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
    # expand_new_xyz = fluid.layers.unsqueeze(new_xyz, axes=[2])
    expand_new_xyz = unsqueeze(new_xyz, axis=2)
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
    # grouped_xyz = fluid.layers.unsqueeze(xyz, axes=[2]) #[-1,128,1,3]
    grouped_xyz = unsqueeze(xyz, axis=2) #[-1,128,1,3]
    #print("grouped_xyz:",grouped_xyz)
    if features is not None:
        # grouped_features = fluid.layers.unsqueeze(features, axes=[2]) # [-1,128,1,640]
        grouped_features = unsqueeze(features, axis=2)# [-1,128,1,640]
	#print("grouped_features:",grouped_features)
        return fluid.layers.concat([grouped_xyz, grouped_features], axis=-1) if use_xyz else grouped_features
    else:
        return grouped_xyz


class ConvBN(fluid.dygraph.Layer):
    def __init__(self,
                 name_scope,
                 num_filters,
                 filter_size=1,
                 stride=1,
                 padding=0,
                 dilation=1,
                 act="relu",
                 bn=True):
        super(ConvBN,self).__init__(name_scope)

        param_attr = ParamAttr(name='conv_weight',
                               initializer=fluid.initializer.Constant(1.376))
        if bn:
            bias_attr = None
            self.BN = BatchNorm(self.full_name(),
                                num_channels=num_filters,
                                act=act,
                                param_attr=ParamAttr(initializer=fluid.initializer.Constant(2.673),
                                                     name="scale"),
                                bias_attr=ParamAttr(initializer=fluid.initializer.Constant(1.467),
                                                    name="offset"),
                                moving_mean_name="mean",
                                moving_variance_name="var")
        else:
            bias_attr = ParamAttr(initializer=fluid.initializer.Constant(0.213),
                                  name='conv_bias')
        self.conv = Conv2D(
            self.full_name(),
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            use_cudnn=True,
            param_attr=param_attr,
            bias_attr=bias_attr,
            act=act if not bn else None)
        self.bn = bn

    def forward(self, input):
        out = self.conv(input)
        if self.bn:
            out = self.BN(out)
        return out

    def set_bn_momentum(self, momentum):
        if self.bn:
            self.BN._momentum = momentum


class FCBN(fluid.dygraph.Layer):
    def __init__(self,
                 name_scope,
                 out_channel,
                 act="relu",
                 bn=False):
        super(FCBN,self).__init__(name_scope)

        param_attr = ParamAttr(name="fc_weight",
                               initializer=fluid.initializer.Constant(2.4))
        if not bn:
            bias_attr = ParamAttr(name="fc_bias",
                                  initializer=fluid.initializer.Constant(1.4))
        else:
            bias_attr = None
            self.BN = BatchNorm(self.full_name(),
                                num_channels=out_channel,
                                act=act,
                                param_attr=ParamAttr(initializer=fluid.initializer.Constant(2.673)),
                                bias_attr=ParamAttr(initializer=fluid.initializer.Constant(1.467)))
        self.fc = FC(self.full_name(),
                     size=out_channel,
                     param_attr=param_attr,
                     bias_attr=bias_attr,
                     act=act if not bn else None)
        self.bn = bn

    def forward(self, inputs):
        out = self.fc(inputs)
        if self.bn:
            out = self.BN(out)
        return out

    def set_bn_momentum(self, momentum):
        if bn:
            self.BN._momentum = momentum


class MLP(fluid.dygraph.Layer):
    def __init__(self,
                 name_scope,
                 out_channels_list,
                 bn=True,
                 act="relu"):
        super(MLP,self).__init__(name_scope)

        self.build_conv_list = []

        for i,out_channels in enumerate(out_channels_list):
            conv_block = self.add_sublayer(
                "conv_block_%d" % (i),
                ConvBN(self.full_name(),out_channels,bn=bn,act=act))
            self.build_conv_list.append(conv_block)

    def forward(self, inputs):
        out = inputs
        for conv_block in self.build_conv_list:
            out = conv_block(out)
        return out

    def set_bn_momentum(self, momentum):
        for conv_block in self.build_conv_list:
            conv_block.set_bn_momentum(momentum)


class Pointnet_SA_Module_MSG(fluid.dygraph.Layer):
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
        use_xyz (bool): whether use xyz coordiantes features

    Returns:
        new_xyz (Variable): centriods features with shape [B, npoint, 3]
        out (Variable): features with shape [B, npoint, \sum_i{mlps[i][-1]}]
    """
    def __init__(self,
                 name_scope,
                 npoint=None,
                 radiuss=[],
                 nsamples=[],
                 mlps=[],
                 bn=True,
                 use_xyz=True):
        super(Pointnet_SA_Module_MSG,self).__init__(name_scope)

        self.build_mlp_list = []

        for i,mlp in enumerate(mlps):
            mlp_block = self.add_sublayer(
                "mlp_block_%d" % (i),
                MLP(self.full_name(),mlp,bn))
            self.build_mlp_list.append(mlp_block)
        self.npoint = npoint
        self.radiuss = radiuss
        self.nsamples = nsamples
        self.use_xyz = use_xyz
        self.mlps = mlps

    def forward(self, xyz, feature):
        assert len(self.radiuss) == len(self.nsamples) == len(self.mlps), \
            "radiuss, nsamples, mlps length should be same"
        farthest_idx = fluid.layers.farthest_point_sampling(xyz, self.npoint)
        new_xyz = fluid.layers.gather_point(xyz, farthest_idx)

        out = None
        if feature is not None:
            outs = []
            for i , (radius, nsample, mlp) in enumerate(zip(self.radiuss,self.nsamples,self.build_mlp_list)):
                out = query_and_group(xyz, new_xyz, radius, nsample, feature,self.use_xyz) \
                    if self.npoint is not None else group_all(xyz, feature, self.use_xyz)
                out = fluid.layers.transpose(out, perm=[0, 3, 1, 2])
                out = mlp(out)
		if self.npoint is None:
		   out = fluid.layers.transpose(out,perm=[0,1,3,2])
                out = fluid.layers.pool2d(out, pool_size=[1, out.shape[3]], pool_type='max')
                # out = fluid.layers.squeeze(out, axes=[-1])
                out = squeeze(out, axis=-1)
                outs.append(out)
            out = fluid.layers.concat(outs,axis=1)
            out = fluid.layers.transpose(out,perm=[0,2,1])
        return(new_xyz,out)

    def set_bn_momentum(self, momentum):
        for mlp in self.build_mlp_list:
            mlp.set_bn_momentum(momentum)


class Pointnet_FP_module(fluid.dygraph.Layer):
    """
    PointNet Feature Propagation Module

    Args:
        unknown (Variable): unknown xyz coordiantes features with shape [B, N, 3]
        known (Variable): known xyz coordiantes features with shape [B, M, 3]
        unknown_feats (Variable): unknown features with shape [B, N, C1] to be propagated to
        known_feats (Variable): known features with shape [B, M, C2] to be propagated from
        mlp ([int32]): out_channels_list
        bn (bool): whether perform batch norm after conv2d

    Returns:
        new_features (Variable): new features with shape [B, N, mlp[-1]]
    """
    def __init__(self,
                 name_scope,
                 mlp,
                 bn=True):
        super(Pointnet_FP_module, self).__init__(name_scope)
        self.MLP = MLP(self.full_name(),mlp,bn)
        self.mlp = mlp
        self.bn = bn

    def forward(self, unknown,known,unknown_feats,known_feats):
        if known is None:
            interp_feats = fluid.layers.expand()
        else:
            dist, idx = fluid.layers.three_nn(unknown, known, eps=0)
            dist = fluid.layers.sqrt(dist)
            ones = fluid.layers.fill_constant_batch_size_like(dist,dist.shape,dist.dtype,1)
            dist_recip = ones / (dist + 1e-8)
            norm = fluid.layers.reduce_sum(dist_recip,dim=-1,keep_dim=True)
            weight = dist_recip / norm
            interp_feats = fluid.layers.three_interp(known_feats,weight,idx)
        new_features = interp_feats if unknown_feats is None else \
            fluid.layers.concat([interp_feats, unknown_feats], axis=-1)
        new_features = fluid.layers.transpose(new_features, perm=[0,2,1])
        # new_features = fluid.layers.unsqueeze(new_features, axes=[-1])
        new_features = unsqueeze(new_features, axis=-1)
        new_features = self.MLP(new_features)
        # new_features = fluid.layers.squeeze(new_features, axes=[-1])
        new_features = squeeze(new_features, axis=-1)
        new_features = fluid.layers.transpose(new_features, perm=[0,2,1])
        return new_features

    def set_bn_momentum(self, momentum):
        self.MLP.set_bn_momentum(momentum)

if __name__ == "__main__":
    xyz = fluid.layers.data(name='xyz', shape=[9, 3], dtype='float32')
    xyz_feats = fluid.layers.data(name='xyz_feats', shape=[12, 18], dtype='float32')
    pointnet_sa = Pointnet_SA_Module_MSG("sa_module_msg", 4, [0.8, 1.6], [6, 3], [[3, 6], [6, 9]])
    new_xyz, out = pointnet_sa(xyz, xyz_feats)

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
    # pointnet_fp = Pointnet_FP_module("fp_module_msg", [6])
    # new_features = pointnet_fp(unknown, known, unknown_feats, known_feats)
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
