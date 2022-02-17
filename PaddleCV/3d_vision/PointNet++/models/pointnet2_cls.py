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
Contains PointNet++ classification models
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.initializer import Constant
from .pointnet2_modules import *

__all__ = ["PointNet2ClsSSG", "PointNet2ClsMSG"]


class PointNet2Cls(object):
    def __init__(self, num_classes, num_points, use_xyz=True):
        self.num_classes = num_classes
        self.num_points = num_points
        self.use_xyz = use_xyz
        self.out_feature = None
        self.loader = None
        self.model_config()

    def model_config(self):
        self.SA_confs = []

    def build_input(self):
        self.xyz = fluid.data(name='xyz',
                              shape=[None, self.num_points, 3],
                              dtype='float32',
                              lod_level=0)
        self.label = fluid.data(name='label',
                                shape=[None, 1],
                                dtype='int64',
                                lod_level=0)
        self.loader = fluid.io.DataLoader.from_generator(
                feed_list=[self.xyz, self.label],
                capacity=64,
                use_double_buffer=True,
                iterable=False)
        self.feed_vars = [self.xyz, self.label]

    def build_model(self, bn_momentum=0.99):
        self.build_input()

        xyz, feature = self.xyz, None
        for i, SA_conf in enumerate(self.SA_confs):
            xyz, feature = pointnet_sa_module(
                    xyz=xyz,
                    feature=feature,
                    bn_momentum=bn_momentum,
                    use_xyz=self.use_xyz,
                    name="sa_{}".format(i),
                    **SA_conf)

        out = fluid.layers.squeeze(feature, axes=[-1])
        out = fc_bn(out, out_channels=512, bn=True, bn_momentum=bn_momentum, name="fc_1")
        out = fluid.layers.dropout(out, 0.5, dropout_implementation="upscale_in_train")
        out = fc_bn(out, out_channels=256, bn=True, bn_momentum=bn_momentum, name="fc_2")
        out = fluid.layers.dropout(out, 0.5, dropout_implementation="upscale_in_train")
        out = fc_bn(out, out_channels=self.num_classes, act=None, name="fc_3")
        pred = fluid.layers.softmax(out)

        # calc loss
        self.loss = fluid.layers.cross_entropy(pred, self.label)
        self.loss = fluid.layers.reduce_mean(self.loss)

        # calc acc
        pred = fluid.layers.reshape(pred, shape=[-1, self.num_classes])
        label = fluid.layers.reshape(self.label, shape=[-1, 1])
        self.acc1 = fluid.layers.accuracy(pred, label, k=1)

    def get_feeds(self):
        return self.feed_vars

    def get_outputs(self):
        return {"loss": self.loss, "accuracy": self.acc1}

    def get_loader(self):
        return self.loader


class PointNet2ClsSSG(PointNet2Cls):
    def __init__(self, num_classes, num_points, use_xyz=True):
        super(PointNet2ClsSSG, self).__init__(num_classes, num_points, use_xyz)

    def model_config(self):
        self.SA_confs = [
            {
                "npoint": 512,
                "radiuss": [0.2],
                "nsamples": [64],
                "mlps": [[64, 64, 128]],
            },
            {
                "npoint": 128,
                "radiuss": [0.4],
                "nsamples": [64],
                "mlps": [[128, 128, 256]],
            },
            {
                "npoint":None,
		"radiuss": [None],
		"nsamples":[None],
		"mlps": [[256, 512, 1024]],
            },
        ]


class PointNet2ClsMSG(PointNet2Cls):
    def __init__(self, num_classes, num_points, use_xyz=True):
        super(PointNet2ClsMSG, self).__init__(num_classes, num_points, use_xyz)

    def model_config(self):
        self.SA_confs = [
            {
                "npoint": 512,
                "radiuss": [0.1, 0.2, 0.4],
                "nsamples": [16, 32, 128],
                "mlps": [[32, 32, 64],
                         [64, 64, 128],
                         [64,96,128]],
            },
            {
                "npoint": 128,
                "radiuss": [0.2, 0.4, 0.8],
                "nsamples": [32, 64, 128],
                "mlps": [[64, 64, 128],
                         [128, 128, 256],
                         [128,128,256]],
            },
            {
                "npoint":None,
		"radiuss": [None],
		"nsamples":[None],
		"mlps": [[256, 512, 1024]],
            },
        ]


