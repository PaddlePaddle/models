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
Contains PointNet++ MSG classification models
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.initializer import Constant
from pointnet2_modules_dy import *


class PointNet2MSGCls(fluid.dygraph.Layer):
    def __init__(self, name_scope,num_classes=13, use_xyz=True):
        super(PointNet2MSGCls,self).__init__(name_scope)
        self.num_classes = num_classes
        self.use_xyz = use_xyz
        self.sa_module_msg_0 = Pointnet_SA_Module_MSG(self.full_name(),
                                                  npoint=512,
                                                  radiuss=[0.1,0.2,0.4],
                                                  nsamples=[16,32,128],
                                                  mlps=[[32,32,64],[64,64,128],[64,96,128]])
        self.sa_module_msg_1 = Pointnet_SA_Module_MSG(self.full_name(),
                                                      npoint=128,
                                                      radiuss=[0.2,0.4,0.8],
                                                      nsamples=[32,64,128],
                                                      mlps=[[64,64,128],[128,128,256],[128,128,256]])
        self.sa_module = Pointnet_SA_Module_MSG(self.full_name(),radiuss=[None],nsamples=[None],mlps=[[256,512,1024]])
        self.fc_0 = FCBN(self.full_name(),out_channel=512,bn=True)
        self.fc_1 = FCBN(self.full_name(),out_channel=256,bn=True)
        self.fc_2 = FCBN(self.full_name(),out_channel=self.num_classes,act=None)
    def forward(self, xyz,feature,label):
        xyz,feature = self.sa_module_msg_0(xyz,feature)
        xyz,feature = self.sa_module_msg_1(xyz,feature)
        xyz,feature = self.sa_module(xyz,feature)
	print (feature)
        out = fluid.layers.transpose(feature, perm=[0, 2, 1])
        out = fluid.layers.squeeze(out, axes=[-1])
        # FC layer
        out = self.fc_0(out)
        #out = fluid.layers.dropout(out,0.5)
        out = self.fc_1(out)
        #out = fluid.layers.dropout(out,0.5)
        out = self.fc_2(out)
        pred = fluid.layers.softmax(out)

        # calc loss
        loss = fluid.layers.cross_entropy(pred,label)
        loss = fluid.layers.reduce_mean(loss)

        # calc acc
        pred = fluid.layers.reshape(pred,shape=[-1,self.num_classes])
        label = fluid.layers.reshape(label,shape=[-1,1])
        acc1 = fluid.layers.accuracy(pred,label,k=1)
        acc2 = fluid.layers.accuracy(pred,label,k=5)
        return loss,acc1,acc2



if __name__ == "__main__":
    num_classes = 13
    pointnet_cls = PointNet2MSGCls("pointnet2_cls",num_classes=13)
    xyz = fluid.layers.data(name='xyz', shape=[32, 3], dtype='float32', lod_level=0)
    feature = fluid.layers.data(name='feature', shape=[32, 6], dtype='float32', lod_level=0)
    label = fluid.layers.data(name='label', shape=[1], dtype='int64', lod_level=0)

    loss,_,_= pointnet_cls(xyz,feature,label)
    opt = fluid.optimizer.Adam(learning_rate=1e-2)
    opt.minimize(loss)


    place = fluid.CUDAPlace(0)
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    np.random.seed(1333)
    xyz_np = np.random.uniform(-100, 100, (8, 32, 3)).astype('float32')
    feature_np = np.random.uniform(-100, 100, (8, 32, 6)).astype('float32')
    label_np = np.random.uniform(0, num_classes, (8, 1)).astype('int64')
    #print("xyz", xyz_np)
    #print("feaure", feature_np)
    #print("label", label_np)
    for i in range(5):
        ret = exe.run(fetch_list=["transpose_10.tmp_0@GRAD",loss.name], feed={'xyz': xyz_np, 'feature': feature_np, 'label': label_np})
	print("loss:",ret)
    #ret = exe.run(fetch_list=["relu_0.tmp_0","relu_1.tmp_0","relu_2.tmp_0", outs[0].name, outs[1].name], feed={'xyz': xyz_np, 'feature': feature_np, 'label': label_np})
    #print(ret)
    # print("ret0", ret[0].shape, ret[0])
    # ret[0].tofile("out.data")
