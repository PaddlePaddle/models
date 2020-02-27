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
        self.loader = None

    def build_inputs(self):
        self.inputs = OrderedDict()

        if self.cfg.RPN.ENABLED:
            self.inputs['sample_id'] = fluid.data(name='sample_id', shape=[None, 1], dtype='int32')
            self.inputs['pts_input'] = fluid.data(name='pts_input', shape=[None, self.num_points, 3], dtype='float32')
            self.inputs['pts_rect'] = fluid.data(name='pts_rect', shape=[None, self.num_points, 3], dtype='float32')
            self.inputs['pts_features'] = fluid.data(name='pts_features', shape=[None, self.num_points, 1], dtype='float32')
            self.inputs['rpn_cls_label'] = fluid.data(name='rpn_cls_label', shape=[None, self.num_points], dtype='int32')
            self.inputs['rpn_reg_label'] = fluid.data(name='rpn_reg_label', shape=[None, self.num_points, 7], dtype='float32')
            self.inputs['gt_boxes3d'] = fluid.data(name='gt_boxes3d', shape=[None, 7], lod_level=1, dtype='float32')

        if self.cfg.RCNN.ENABLED:
            if self.cfg.RCNN.ROI_SAMPLE_JIT:
                self.inputs['sample_id'] = fluid.data(name='sample_id', shape=[1], dtype='int32')
                self.inputs['rpn_xyz'] = fluid.data(name='rpn_xyz', shape=[self.num_points, 3], dtype='float32')
                self.inputs['rpn_features'] = fluid.data(name='rpn_features', shape=[self.num_points, 128], dtype='float32')
                self.inputs['rpn_intensity'] = fluid.data(name='rpn_intensity', shape=[self.num_points], dtype='float32')
                self.inputs['seg_mask'] = fluid.data(name='seg_mask', shape=[self.num_points], dtype='float32')
                self.inputs['roi_boxes3d'] = fluid.data(name='roi_boxes3d', shape=[None, None, 7], dtype='float32', lod_level=0)
                self.inputs['pts_depth'] = fluid.data(name='pts_depth', shape=[self.num_points], dtype='float32')
                self.inputs['gt_boxes3d'] = fluid.data(name='gt_boxes3d', shape=[None, None, 7], dtype='float32', lod_level=0)
            else:
                self.inputs['sample_id'] = fluid.data(name='sample_id', shape=[None], dtype='int32')
                self.inputs['pts_input'] = fluid.data(name='pts_input', shape=[None, 512, 133], dtype='float32')
                self.inputs['pts_feature'] = fluid.data(name='pts_feature', shape=[None, 512, 128], dtype='float32')
                self.inputs['roi_boxes3d'] = fluid.data(name='roi_boxes3d', shape=[None,7], dtype='float32')
                if self.is_train:
                    self.inputs['cls_label'] = fluid.data(name='cls_label', shape=[None], dtype='float32')
                    self.inputs['reg_valid_mask'] = fluid.data(name='reg_valid_mask', shape=[None], dtype='float32')
                    self.inputs['gt_boxes3d_ct'] = fluid.data(name='gt_boxes3d_ct', shape=[None, 7], dtype='float32')
                    self.inputs['gt_of_rois'] = fluid.data(name='gt_of_rois', shape=[None, 7], dtype='float32')
                else:
                    self.inputs['roi_scores'] = fluid.data(name='roi_scores', shape=[None], dtype='float32')
                    self.inputs['gt_iou'] = fluid.data(name='gt_iou', shape=[None], dtype='float32')
                    self.inputs['gt_boxes3d'] = fluid.data(name='gt_boxes3d', shape=[None, None, 7], dtype='float32', lod_level=0)
                

        self.loader = fluid.io.DataLoader.from_generator(
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

    def get_loader(self):
        return self.loader
        
