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
from models.rcnn import RCNN


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

        if self.cfg.RPN.ENABLED:
            self.inputs['sample_id'] = fluid.layers.data(name='sample_id', shape=[1], dtype='int32')
            self.inputs['pts_input'] = fluid.layers.data(name='pts_input', shape=[self.num_points, 3], dtype='float32')
            self.inputs['pts_rect'] = fluid.layers.data(name='pts_rect', shape=[self.num_points, 3], dtype='float32')
            self.inputs['pts_features'] = fluid.layers.data(name='pts_features', shape=[self.num_points, 1], dtype='float32')
            self.inputs['rpn_cls_label'] = fluid.layers.data(name='rpn_cls_label', shape=[self.num_points], dtype='int32')
            self.inputs['rpn_reg_label'] = fluid.layers.data(name='rpn_reg_label', shape=[self.num_points, 7], dtype='float32')
            self.inputs['gt_boxes3d'] = fluid.layers.data(name='gt_boxes3d', shape=[7], lod_level=1, dtype='float32')

        if self.cfg.RCNN.ENABLED:
            if self.cfg.RCNN.ROI_SAMPLE_JIT:
                self.inputs['sample_id'] = fluid.layers.data(name='sample_id', shape=[1], dtype='int32', append_batch_size=False)
                self.inputs['rpn_xyz'] = fluid.layers.data(name='rpn_xyz', shape=[self.num_points, 3], dtype='float32', append_batch_size=False)
                self.inputs['rpn_features'] = fluid.layers.data(name='rpn_features', shape=[self.num_points,128], dtype='float32', append_batch_size=False)
                self.inputs['rpn_intensity'] = fluid.layers.data(name='rpn_intensity', shape=[self.num_points], dtype='float32', append_batch_size=False)
                self.inputs['seg_mask'] = fluid.layers.data(name='seg_mask', shape=[self.num_points], dtype='float32', append_batch_size=False)
                self.inputs['roi_boxes3d'] = fluid.layers.data(name='roi_boxes3d', shape=[-1, -1, 7], dtype='float32', append_batch_size=False, lod_level=0)
                self.inputs['pts_depth'] = fluid.layers.data(name='pts_depth', shape=[self.num_points], dtype='float32', append_batch_size=False)
                self.inputs['gt_boxes3d'] = fluid.layers.data(name='gt_boxes3d', shape=[-1, -1, 7], dtype='float32', append_batch_size=False, lod_level=0)
            else:
                self.inputs['sample_id'] = fluid.layers.data(name='sample_id', shape=[-1], dtype='int32', append_batch_size=False)
                self.inputs['pts_input'] = fluid.layers.data(name='pts_input', shape=[-1,512,133], dtype='float32', append_batch_size=False)
                self.inputs['pts_feature'] = fluid.layers.data(name='pts_feature', shape=[-1,512,128], dtype='float32', append_batch_size=False)
                self.inputs['roi_boxes3d'] = fluid.layers.data(name='roi_boxes3d', shape=[-1,7], dtype='float32', append_batch_size=False)
                if self.is_train:
                    self.inputs['cls_label'] = fluid.layers.data(name='cls_label', shape=[-1], dtype='float32', append_batch_size=False)
                    self.inputs['reg_valid_mask'] = fluid.layers.data(name='reg_valid_mask', shape=[-1], dtype='float32', append_batch_size=False)
                    self.inputs['gt_boxes3d_ct'] = fluid.layers.data(name='gt_boxes3d_ct', shape=[-1,7], dtype='float32', append_batch_size=False)
                    self.inputs['gt_of_rois'] = fluid.layers.data(name='gt_of_rois', shape=[-1,7], dtype='float32', append_batch_size=False)
                else:
                    self.inputs['roi_scores'] = fluid.layers.data(name='roi_scores', shape=[-1,], dtype='float32', append_batch_size=False)
                    self.inputs['gt_iou'] = fluid.layers.data(name='gt_iou', shape=[-1], dtype='float32', append_batch_size=False)
                    self.inputs['gt_boxes3d'] = fluid.layers.data(name='gt_boxes3d', shape=[-1,-1,7], dtype='float32', append_batch_size=False, lod_level=0)
                

        self.pyreader = fluid.io.PyReader(
                feed_list=list(self.inputs.values()),
                capacity=64,
                use_double_buffer=True,
                iterable=False)

    def build(self):
        self.build_inputs()
        if self.cfg.RPN.ENABLED:
            self.rpn = RPN(self.cfg, self.batch_size, self.use_xyz,
                           self.mode, self.prog)
            self.rpn.build(self.inputs)
            self.rpn_outputs = self.rpn.get_outputs()
            self.outputs = self.rpn_outputs
        
        if self.cfg.RCNN.ENABLED:
            self.rcnn = RCNN(self.cfg, 1, self.batch_size, self.mode)
            self.rcnn.build_model(self.inputs)
            self.outputs = self.rcnn.get_outputs()
        
        if self.mode == 'TRAIN':
            if self.cfg.RPN.ENABLED:
                self.outputs['rpn_loss'], self.outputs['rpn_loss_cls'], \
                        self.outputs['rpn_loss_reg'] = self.rpn.get_loss()
            if self.cfg.RCNN.ENABLED:
                self.outputs['rcnn_loss'], self.outputs['rcnn_loss_cls'], \
                        self.outputs['rcnn_loss_reg'] = self.rcnn.get_loss()
            self.outputs['loss'] = self.outputs.get('rpn_loss', 0.) \
                                 + self.outputs.get('rcnn_loss', 0.)

    def get_feeds(self):
        return list(self.inputs.keys())

    def get_outputs(self):
        return self.outputs

    def get_loss(self):
        rpn_loss, _, _ = self.rpn.get_loss()
        rcnn_loss, _, _ = self.rcnn.get_loss()
        return rpn_loss + rcnn_loss

    def get_pyreader(self):
        return self.pyreader
        
