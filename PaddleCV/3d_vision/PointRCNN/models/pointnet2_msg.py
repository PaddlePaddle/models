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
from models.pointnet2_modules import *

__all__ = ["PointNet2MSG"]


class PointNet2MSG(object):
    def __init__(self, cfg, xyz, feature=None, use_xyz=True):
        self.cfg = cfg
        self.xyz = xyz
        self.feature = feature
        self.use_xyz = use_xyz
        self.model_config()

    def model_config(self):
        self.SA_confs = []
        for i in range(self.cfg.RPN.SA_CONFIG.NPOINTS.__len__()):
            self.SA_confs.append({
                "npoint": self.cfg.RPN.SA_CONFIG.NPOINTS[i],
                "radiuss": self.cfg.RPN.SA_CONFIG.RADIUS[i],
                "nsamples": self.cfg.RPN.SA_CONFIG.NSAMPLE[i],
                "mlps": self.cfg.RPN.SA_CONFIG.MLPS[i],
                })

        self.FP_confs = []
        for i in range(self.cfg.RPN.FP_MLPS.__len__()):
            self.FP_confs.append({"mlp": self.cfg.RPN.FP_MLPS[i]})

    def build(self, bn_momentum=0.95):
        xyzs, features = [self.xyz], [self.feature]
        xyzi, featurei = self.xyz, self.feature
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
                    name="fp_{}".format(i + len(self.FP_confs)),
                    **self.FP_confs[i])

        return xyzs[0], features[0]

