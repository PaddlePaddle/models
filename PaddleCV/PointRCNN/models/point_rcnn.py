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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from collections import OrderedDict

import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.initializer import Constant

from models.rpn import RPN


__all__ = ["PointRCNN"]


class PointRCNN(object):
    def __init__(self, cfg, batch_size, use_xyz=True, mode='TRAIN', prog=None):
        self.cfg = cfg
        self.batch_size = batch_size
        self.use_xyz = use_xyz
        self.mode = mode
        self.is_train = mode == 'TRAIN'
        self.num_points = self.cfg.RPN.NUM_POINTS
        self.prog = prog
        self.inputs = None
        self.pyreader = None

    def build_inputs(self):
        self.inputs = OrderedDict()
        self.inputs['sample_id'] = fluid.layers.data(name='sample_id', shape=[1], dtype='int32')
        self.inputs['pts_input'] = fluid.layers.data(name='pts_input', shape=[self.num_points, 3], dtype='float32')
        self.inputs['pts_rect'] = fluid.layers.data(name='pts_rect', shape=[self.num_points, 3], dtype='float32')
        self.inputs['pts_features'] = fluid.layers.data(name='pts_features', shape=[self.num_points, 1], dtype='float32')
        self.inputs['rpn_cls_label'] = fluid.layers.data(name='rpn_cls_label', shape=[self.num_points], dtype='int32')
        self.inputs['rpn_reg_label'] = fluid.layers.data(name='rpn_reg_label', shape=[self.num_points, 7], dtype='float32')
        self.inputs['gt_boxes3d'] = fluid.layers.data(name='gt_boxes3d', shape=[7], lod_level=1, dtype='float32')
        self.pyreader = fluid.io.PyReader(
                feed_list=self.inputs.values(),
                capacity=64,
                use_double_buffer=True,
                iterable=False)

    def build(self):
        self.build_inputs()
        if self.cfg.RPN.ENABLED:
            self.rpn = RPN(self.cfg, self.batch_size, self.use_xyz, self.mode, self.prog)
            self.rpn.build(self.inputs)
            self.rpn_outpus = self.rpn.get_outputs()
        if self.cfg.RCNN.ENABLED:
            self.inputs.update(self.rpn_outpus)
            # self.rcnn = RCNN()
        self.outputs = self.rpn_outpus
        
        if self.cfg.RPN.ENABLED:
            self.outputs['rpn_loss'] = self.rpn.get_loss()[0]
        if self.cfg.RCNN.ENABLED:
            self.outputs['rcnn_loss'] = self.rcnn.get_loss()
        self.outputs['loss'] = self.outputs.get('rpn_loss', 0.) \
                             + self.outputs.get('rcnn_loss', 0.)

    def get_feeds(self):
        return self.inputs.keys()

    def get_outputs(self):
        # ret_keys = ['loss', 'rpn_loss', 'rcnn_loss', ]
        return self.outputs

    def get_loss(self):
        rpn_loss, _, _ = self.rpn.get_loss()
        rcnn_loss = 0.
        return rpn_loss + rcnn_loss

    def get_pyreader(self):
        return self.pyreader
        
