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
from pointnet2_modules import *


class PointNet2MSGCls(object):
    def __init__(self, num_classes, use_xyz=True):
        self.num_classes = num_classes
        self.use_xyz = use_xyz
        self.out_feature = None
        self.pyreader = None
        self.model_config()

    def model_config(self):
        self.SA_confs = []

    def build_input(self):
        self.xyz = fluid.layers.data(name='xyz', shape=[32, 3], dtype='float32', lod_level=0)
        self.feature = fluid.layers.data(name='feature', shape=[32, 6], dtype='float32', lod_level=0)
        self.label = fluid.layers.data(name='label', shape=[1], dtype='int64', lod_level=0)
        # self.py_reader = fluid.io.PyReader(
        #         feed_list=[self.xyz, self.feature, self.label],
        #         capacity=64,
        #         use_double_buffer=True,
        #         iterable=False)

    def build_model(self):
        self.build_input()

        xyz, feature = self.xyz, self.feature # self.feature:[-1,32,6]
        for i, SA_conf in enumerate(self.SA_confs):
            xyz, feature = pointnet_sa_module(
                    xyz=xyz,
                    feature=feature,
                    use_xyz=self.use_xyz,
                    name="sa_{}".format(i),
                    **SA_conf)
	#feature:[-1.1.1024]
	#transpose_10.tmp_0
	out = fluid.layers.transpose(feature,perm=[0,2,1])
        out = fluid.layers.squeeze(out,axes=[-1])

	out = fc_bn(out,out_channels=512,bn=True,name="fc_1")
        #out = fluid.layers.dropout(out, 0.5)
        out = fc_bn(out,out_channels=256,bn=True,name="fc_2")
        #out = fluid.layers.dropout(out, 0.5)
        out = fc_bn(out,out_channels=self.num_classes,act=None,name="fc_3")

	#softmax
        #self.pred_ = fluid.layers.softmax(out)

        # calc loss
        #self.loss = fluid.layers.cross_entropy(self.pred_, self.label)
        #self.loss = fluid.layers.reduce_mean(self.loss)

	#sigmoid

	label_onehot = fluid.layers.one_hot(self.label, depth=self.num_classes)
	label_float = fluid.layers.cast(label_onehot, dtype='float32')
	self.loss = fluid.layers.sigmoid_cross_entropy_with_logits(out,label_float)
	self.loss = fluid.layers.reduce_mean(self.loss)

        # calc acc
        pred = fluid.layers.reshape(out, shape=[-1, self.num_classes])
        label = fluid.layers.reshape(self.label, shape=[-1, 1])
        self.acc1 = fluid.layers.accuracy(pred, label, k=1)
        self.acc5 = fluid.layers.accuracy(pred, label, k=5)

    def get_feeds(self):
        return self.feed_vars

    def get_outputs(self):
        return self.loss, self.acc1, self.acc5


class PointNet2CLSMSG(PointNet2MSGCls):
    def __init__(self, num_classes, use_xyz=True):
        super(PointNet2CLSMSG, self).__init__(num_classes, use_xyz)

    def model_config(self):
        self.SA_confs = [
            {
                "npoint": 512,
                "radiuss": [0.1, 0.2, 0.4],
                "nsamples": [16, 32, 128],
                "mlps": [[32, 32, 64], [64, 64, 128], [64,96,128]],
            },
            {
                "npoint": 128,
                "radiuss": [0.2, 0.4, 0.8],
                "nsamples": [32, 64, 128],
                "mlps": [[64, 64, 128], [128, 128, 256], [128,128,256]],
            },
            {
                "npoint":None,
		"radiuss": [None],
		"nsamples":[None],
		"mlps": [[256, 512, 1024]],
            },
        ]



if __name__ == "__main__":
    num_classes = 13
    
    model = PointNet2CLSMSG(num_classes)
    model.build_model()
    loss,_,_ = model.get_outputs()
    opt = fluid.optimizer.AdamOptimizer(learning_rate=3e-2)
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
    print("label", label_np)
    for i in range(5):
        ret = exe.run(fetch_list=[loss.name], feed={'xyz': xyz_np, 'feature': feature_np, 'label': label_np})
	print(ret)
    #ret = exe.run(fetch_list=["relu_0.tmp_0","relu_1.tmp_0","relu_2.tmp_0", outs[0].name, outs[1].name], feed={'xyz': xyz_np, 'feature': feature_np, 'label': label_np})
    #print(ret)
    # print("ret0", ret[0].shape, ret[0])
    # ret[0].tofile("out.data")
