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
from .pointnet2_modules import *

__all__ = ["PointNet2SemSegSSG", "PointNet2SemSegMSG"]


class PointNet2SemSeg(object):
    def __init__(self, num_classes, num_points, use_xyz=True):
        self.num_classes = num_classes
        self.num_points = num_points
        self.use_xyz = use_xyz
        self.feed_vars = []
        self.out_feature = None
        self.loader = None
        self.model_config()

    def model_config(self):
        self.SA_confs = []
        self.FP_confs = []

    def build_input(self):
        self.xyz = fluid.data(name='xyz',
                              shape=[None, self.num_points, 3],
                              dtype='float32',
                              lod_level=0)
        self.feature = fluid.data(name='feature',
                                  shape=[None, self.num_points, 6],
                                  dtype='float32',
                                  lod_level=0)
        self.label = fluid.data(name='label',
                                shape=[None, self.num_points, 1],
                                dtype='int64',
                                lod_level=0)
        self.loader = fluid.io.DataLoader.from_generator(
                feed_list=[self.xyz, self.feature, self.label],
                capacity=64,
                use_double_buffer=True,
                iterable=False)
        self.feed_vars = [self.xyz, self.feature, self.label]

    def build_model(self, bn_momentum=0.99):
        self.build_input()

        xyzs, features = [self.xyz], [self.feature]
        xyzi, featurei = xyzs[-1], fluid.layers.transpose(self.feature, perm=[0, 2, 1])
        for i, SA_conf in enumerate(self.SA_confs):
            xyzi, featurei = pointnet_sa_module(
                    xyz=xyzi,
                    feature=featurei,
                    bn_momentum=bn_momentum,
                    use_xyz=self.use_xyz,
                    name="sa_{}".format(i),
                    **SA_conf)
            xyzs.append(xyzi)
            features.append(fluid.layers.transpose(featurei, perm=[0, 2, 1]))
        for i in range(-1, -(len(self.FP_confs) + 1), -1):
            features[i - 1] = pointnet_fp_module(
                    unknown=xyzs[i - 1],
                    known=xyzs[i],
                    unknown_feats=features[i - 1],
                    known_feats=features[i],
                    bn_momentum=bn_momentum,
                    name="fp_{}".format(i+len(self.FP_confs)),
                    **self.FP_confs[i])

        out = fluid.layers.transpose(features[0], perm=[0, 2, 1])
        out = fluid.layers.unsqueeze(out, axes=[-1])
        out = conv_bn(out, out_channels=128, bn=True, bn_momentum=bn_momentum, name="output_1")
        out = fluid.layers.dropout(out, 0.5, dropout_implementation="upscale_in_train")
        out = conv_bn(out, out_channels=self.num_classes, bn=False, act=None, name="output_2")
        out = fluid.layers.squeeze(out, axes=[-1])
        out = fluid.layers.transpose(out, perm=[0, 2, 1])
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


class PointNet2SemSegSSG(PointNet2SemSeg):
    def __init__(self, num_classes, use_xyz=True):
        super(PointNet2SemSegSSG, self).__init__(num_classes, use_xyz)

    def model_config(self):
        self.SA_confs = [
            {
                "npoint": 1024,
                "radiuss": [0.1],
                "nsamples": [32],
                "mlps": [[32, 32, 64]],
            },
            {
                "npoint": 256,
                "radiuss": [0.2],
                "nsamples": [32],
                "mlps": [[64, 64, 128]],
            },
            {
                "npoint": 64,
                "radiuss": [0.4],
                "nsamples": [32],
                "mlps": [[128, 128, 256]],
            },
            {
                "npoint": 16,
                "radiuss": [0.8],
                "nsamples": [32],
                "mlps": [[256, 256, 512]],
            },
        ]

        self.FP_confs = [
            {"mlp": [128, 128, 128]},
            {"mlp": [256, 128]},
            {"mlp": [256, 256]},
            {"mlp": [256, 256]},
        ]


class PointNet2SemSegMSG(PointNet2SemSeg):
    def __init__(self, num_classes, use_xyz=True):
        super(PointNet2SemSegMSG, self).__init__(num_classes, use_xyz)

    def model_config(self):
        self.SA_confs = [
            {
                "npoint": 1024,
                "radiuss": [0.05, 0.1],
                "nsamples": [16, 32],
                "mlps": [[16, 16, 32], [32, 32, 64]],
            },
            {
                "npoint": 256,
                "radiuss": [0.1, 0.2],
                "nsamples": [16, 32],
                "mlps": [[64, 64, 128], [64, 96, 128]],
            },
            {
                "npoint": 64,
                "radiuss": [0.2, 0.4],
                "nsamples": [16, 32],
                "mlps": [[128, 196, 256], [128, 196, 256]],
            },
            {
                "npoint": 16,
                "radiuss": [0.4, 0.8],
                "nsamples": [16, 32],
                "mlps": [[256, 256, 512], [256, 384, 512]],
            },
        ]

        self.FP_confs = [
            {"mlp": [128, 128]},
            {"mlp": [256, 256]},
            {"mlp": [512, 512]},
            {"mlp": [512, 512]},
        ]

