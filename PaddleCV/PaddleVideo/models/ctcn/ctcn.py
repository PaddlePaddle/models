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

import paddle
import paddle.fluid as fluid
from paddle.fluid import ParamAttr
import numpy as np

from ..model import ModelBase
from . import fpn_ctcn

import logging
logger = logging.getLogger(__name__)

__all__ = ["CTCN"]


class CTCN(ModelBase):
    """C-TCN model"""

    def __init__(self, name, cfg, mode='train'):
        super(CTCN, self).__init__(name, cfg, mode=mode)
        self.get_config()

    def get_config(self):
        self.img_size = self.get_config_from_sec('MODEL', 'img_size')
        self.concept_size = self.get_config_from_sec('MODEL', 'concept_size')
        self.num_classes = self.get_config_from_sec('MODEL', 'num_classes')
        self.num_anchors = self.get_config_from_sec('MODEL', 'num_anchors')
        self.total_num_anchors = self.get_config_from_sec('MODEL',
                                                          'total_num_anchors')

        self.num_epochs = self.get_config_from_sec('train', 'epoch')
        self.base_learning_rate = self.get_config_from_sec('train',
                                                           'learning_rate')
        self.learning_rate_decay = self.get_config_from_sec(
            'train', 'learning_rate_decay')
        self.l2_weight_decay = self.get_config_from_sec('train',
                                                        'l2_weight_decay')
        self.momentum = self.get_config_from_sec('train', 'momentum')
        self.lr_decay_iter = self.get_config_from_sec('train', 'lr_decay_iter')

    def build_input(self, use_dataloader=True):
        image_shape = [None, 1, self.img_size, self.concept_size]
        loc_shape = [None, self.total_num_anchors, 2]
        cls_shape = [None, self.total_num_anchors]
        fileid_shape = [None, 1]
        self.use_dataloader = use_dataloader
        # set init data to None
        image = None
        loc_targets = None
        cls_targets = None
        fileid = None

        image = fluid.data(name='image', shape=image_shape, dtype='float32')

        feed_list = []
        feed_list.append(image)
        if (self.mode == 'train') or (self.mode == 'valid'):
            loc_targets = fluid.data(
                name='loc_targets', shape=loc_shape, dtype='float32')
            cls_targets = fluid.data(
                name='cls_targets', shape=cls_shape, dtype='int64')
            feed_list.append(loc_targets)
            feed_list.append(cls_targets)
        elif self.mode == 'test':
            loc_targets = fluid.data(
                name='loc_targets', shape=loc_shape, dtype='float32')
            cls_targets = fluid.data(
                name='cls_targets', shape=cls_shape, dtype='int64')
            fileid = fluid.data(
                name='fileid', shape=fileid_shape, dtype='int64')
            feed_list.append(loc_targets)
            feed_list.append(cls_targets)
            feed_list.append(fileid)
        elif self.mode == 'infer':
            # only image feature input when inference
            pass
        else:
            raise NotImplementedError('mode {} not implemented'.format(
                self.mode))

        if use_dataloader:
            assert self.mode != 'infer', \
                        'dataloader is not recommendated when infer, please set use_dataloader to be false.'
            self.dataloader = fluid.io.DataLoader.from_generator(
                feed_list=feed_list, capacity=4, iterable=True)

        self.feature_input = [image]
        self.cls_targets = cls_targets
        self.loc_targets = loc_targets
        self.fileid = fileid

    def create_model_args(self):
        cfg = {}
        cfg['num_anchors'] = self.num_anchors
        cfg['concept_size'] = self.concept_size
        cfg['num_classes'] = self.num_classes
        return cfg

    def build_model(self):
        cfg = self.create_model_args()
        self.videomodel = fpn_ctcn.FPNCTCN(
            num_anchors=cfg['num_anchors'],
            concept_size=cfg['concept_size'],
            num_classes=cfg['num_classes'],
            mode=self.mode)
        loc_preds, cls_preds = self.videomodel.net(input=self.feature_input[0])
        self.network_outputs = [loc_preds, cls_preds]

    def optimizer(self):
        bd = [self.lr_decay_iter]
        base_lr = self.base_learning_rate
        lr_decay = self.learning_rate_decay
        lr = [base_lr, base_lr * lr_decay]
        l2_weight_decay = self.l2_weight_decay
        momentum = self.momentum
        optimizer = fluid.optimizer.Momentum(
            learning_rate=fluid.layers.piecewise_decay(
                boundaries=bd, values=lr),
            momentum=momentum,
            regularization=fluid.regularizer.L2Decay(l2_weight_decay))

        return optimizer

    def loss(self):
        assert self.mode != 'infer', "invalid loss calculationg in infer mode"
        self.loss_ = self.videomodel.loss(self.network_outputs[0],
                                          self.network_outputs[1],
                                          self.loc_targets, self.cls_targets)
        return self.loss_

    def outputs(self):
        loc_preds = self.network_outputs[0]
        cls_preds = fluid.layers.softmax(self.network_outputs[1])
        return [loc_preds, cls_preds]

    def feeds(self):
        if (self.mode == 'train') or (self.mode == 'valid'):
            return self.feature_input + [self.loc_targets, self.cls_targets]
        elif self.mode == 'test':
            return self.feature_input + [
                self.loc_targets, self.cls_targets, self.fileid
            ]
        elif self.mode == 'infer':
            return self.feature_input
        else:
            raise NotImplementedError('mode {} not implemented'.format(
                self.mode))

    def fetches(self):
        if (self.mode == 'train') or (self.mode == 'valid'):
            losses = self.loss()
            fetch_list = [item for item in losses]
        elif self.mode == 'test':
            losses = self.loss()
            preds = self.outputs()
            fetch_list = [item for item in losses] + \
                         [item for item in preds] + \
                         [self.fileid]
        elif self.mode == 'infer':
            preds = self.outputs()
            fetch_list = [item for item in preds]
        else:
            raise NotImplementedError('mode {} not implemented'.format(
                self.mode))
        return fetch_list

    def pretrain_info(self):
        return (None, None)

    def weights_info(self):
        return (
            'CTCN.pdparams',
            'https://paddlemodels.bj.bcebos.com/video_detection/CTCN.pdparams')
