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
Contains PointNet++ SSG/MSG semantic segmentation models
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.initializer import Constant
# from pointnet2_modules_dy import *
from .pointnet2_modules_dy import *

__all__ = ['PointNet2SemSegSSG', 'PointNet2SemSegMSG']


class PointNet2SemSegSSG(fluid.dygraph.Layer):
    def __init__(self, name_scope, num_classes=13, use_xyz=True):
        super(PointNet2SemSegSSG,self).__init__(name_scope)
        self.num_classes = num_classes
        self.use_xyz = use_xyz
        self.sa_module_msg_0 = Pointnet_SA_Module_MSG(self.full_name(),
                                                  npoint=1024,
                                                  radiuss=[0.1],
                                                  nsamples=[32],
                                                  mlps=[[32,32,64]])
        self.sa_module_msg_1 = Pointnet_SA_Module_MSG(self.full_name(),
                                                      npoint=256,
                                                      radiuss=[0.2],
                                                      nsamples=[32],
                                                      mlps=[[64,64,128]])
        self.sa_module_msg_2 = Pointnet_SA_Module_MSG(self.full_name(),
                                                      npoint=64,
                                                      radiuss=[0.4],
                                                      nsamples=[32],
                                                      mlps=[[256, 128, 256]])
        self.sa_module_msg_3 = Pointnet_SA_Module_MSG(self.full_name(),
                                                  npoint=16,
                                                  radiuss=[0.8],
                                                  nsamples=[32],
                                                  mlps=[[256, 256, 512]])
        self.fp_module_0 = Pointnet_FP_module(self.full_name(),
                                              mlp=[128, 128, 128])
        self.fp_module_1 = Pointnet_FP_module(self.full_name(),
                                              mlp=[256, 128])
        self.fp_module_2 = Pointnet_FP_module(self.full_name(),
                                              mlp=[256, 256])
        self.fp_module_3 = Pointnet_FP_module(self.full_name(),
                                              mlp=[256, 256])
        self.conv_bn_0 = ConvBN(self.full_name(),
                              num_filters=128,
                              bn=True)
        self.conv_bn_1 = ConvBN(self.full_name(),
                                num_filters=self.num_classes,bn=False,act=None)
    def forward(self, xyz,feature,label):
        xyz1, feature1 = self.sa_module_msg_0(xyz, feature)
        xyz2, feature2 = self.sa_module_msg_1(xyz1, feature1)
        xyz3, feature3 = self.sa_module_msg_2(xyz2, feature2)
        xyz4, feature4 = self.sa_module_msg_3(xyz3, feature3)

        feature3 = self.fp_module_3(xyz3, xyz4, feature3, feature4)
        feature2 = self.fp_module_2(xyz2, xyz3, feature2, feature3)
        feature1 = self.fp_module_1(xyz1, xyz2, feature1, feature2)
        feature0 = self.fp_module_0(xyz, xyz1, feature, feature1)

        out = fluid.layers.transpose(feature0, perm=[0, 2, 1])
        # out = fluid.layers.unsqueeze(out, axes=[-1])
        out = unsqueeze(out, axis=-1)
        out = self.conv_bn_0(out)
        # out = fluid.layers.dropout(out, 0.5)
        out = self.conv_bn_1(out)
        tmp = out
        # out = fluid.layers.squeeze(out, axes=[-1])
        out = squeeze(out, axis=-1)
        out = fluid.layers.transpose(out, perm=[0, 2, 1])
        pred = fluid.layers.softmax(out)

        #calc loss
        loss = fluid.layers.cross_entropy(pred, label)
        loss = fluid.layers.reduce_mean(loss)

        ##calc acc
        pred_ = fluid.layers.reshape(pred, shape=[-1, self.num_classes])
        label = fluid.layers.reshape(label, shape=[-1, 1])
	#print ("acc label:",pred.shape)
        acc1 = fluid.layers.accuracy(pred_, label, k=1)
        return pred, loss, acc1


class PointNet2SemSegMSG(PointNet2SemSegSSG):
    def __init__(self, name_scope, num_classes=13, use_xyz=True):
        super(PointNet2SemSegMSG, self).__init__(name_scope)
        self.num_classes = num_classes
        self.use_xyz = use_xyz


        self.sa_module_msg_0 = Pointnet_SA_Module_MSG(self.full_name(),
                                                      npoint=1024,
                                                      radiuss=[0.05, 0.1],
                                                      nsamples=[16, 32],
                                                      mlps=[[16, 16, 32], [32, 32, 64]])
        self.sa_module_msg_1 = Pointnet_SA_Module_MSG(self.full_name(),
                                                      npoint=256,
                                                      radiuss=[0.1, 0.2],
                                                      nsamples=[16, 32],
                                                      mlps=[[64,64,128], [64, 96, 128]])
        self.sa_module_msg_2 = Pointnet_SA_Module_MSG(self.full_name(),
                                                      npoint=64,
                                                      radiuss=[0.2, 0.4],
                                                      nsamples=[16, 32],
                                                      mlps=[[128, 196, 256], [128, 196, 256]])
        self.sa_module_msg_3 = Pointnet_SA_Module_MSG(self.full_name(),
                                                  npoint=16,
                                                  radiuss=[0.4, 0.8],
                                                  nsamples=[16, 32],
                                                  mlps=[[256, 256, 512], [256, 384, 512]])
        self.fp_module_0 = Pointnet_FP_module(self.full_name(),
                                              mlp=[128, 128])
        self.fp_module_1 = Pointnet_FP_module(self.full_name(),
                                              mlp=[256, 256])
        self.fp_module_2 = Pointnet_FP_module(self.full_name(),
                                              mlp=[512, 512])
        self.fp_module_3 = Pointnet_FP_module(self.full_name(),
                                              mlp=[512, 512])


if __name__ == "__main__":
    num_classes = 13
    xyz = fluid.layers.data(name='xyz', shape=[32, 3], dtype='float32', lod_level=0)
    feature = fluid.layers.data(name='feature', shape=[32, 6], dtype='float32', lod_level=0)
    label = fluid.layers.data(name='label', shape=[32, 1], dtype='int64', lod_level=0)
    # pointnet_sem_ssg = PointNet2SemSegSSG("pointnet_sem_seg_ssg",num_classes=num_classes)
    pointnet_sem_msg = PointNet2SemSegMSG("pointnet_sem_seg_msg",num_classes=num_classes)

    outs = pointnet_sem_msg(xyz,feature,label)

    place = fluid.CUDAPlace(0)
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    np.random.seed(2333)
    xyz_np = np.random.uniform(-100, 100, (8, 32, 3)).astype('float32')
    feature_np = np.random.uniform(-100, 100, (8, 32, 6)).astype('float32')
    label_np = np.random.uniform(0, num_classes, (8, 32, 1)).astype('int64')
    #print("xyz", xyz_np)
    #print("feaure", feature_np)
    #print("label", label_np)
    ret = exe.run(fetch_list=[out.name for out in outs], feed={'xyz': xyz_np, 'feature': feature_np, 'label': label_np})
    #ret = exe.run(fetch_list=["transpose_17.tmp_0", outs[0].name, outs[1].name], feed={'xyz': xyz_np, 'feature': feature_np, 'label': label_np})
    print(ret[1:])
    print("ret0", ret[0].shape, ret[0])
    # ret[0].tofile("out.data")
