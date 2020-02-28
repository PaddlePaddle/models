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
from . import bsn_net

import logging
logger = logging.getLogger(__name__)

__all__ = ["BsnTem", "BsnPem"]


class BsnTem(ModelBase):
    """BsnTem model"""

    def __init__(self, name, cfg, mode='train'):
        super(BsnTem, self).__init__(name, cfg, mode=mode)
        self.get_config()

    def get_config(self):
        self.tscale = self.get_config_from_sec('MODEL', 'tscale')
        self.feat_dim = self.get_config_from_sec('MODEL', 'feat_dim')
        self.hidden_dim = self.get_config_from_sec('MODEL', 'hidden_dim')

        self.num_epochs = self.get_config_from_sec('train', 'epoch')
        self.base_learning_rate = self.get_config_from_sec('train',
                                                           'learning_rate')
        self.learning_rate_decay = self.get_config_from_sec(
            'train', 'learning_rate_decay')
        self.l2_weight_decay = self.get_config_from_sec('train',
                                                        'l2_weight_decay')
        self.lr_decay_iter = self.get_config_from_sec('train', 'lr_decay_iter')

    def build_input(self, use_dataloader=True):
        feat_shape = [None, self.feat_dim, self.tscale]
        gt_start_shape = [None, self.tscale]
        gt_end_shape = [None, self.tscale]
        gt_action_shape = [None, self.tscale]
        fileid_shape = [None, 1]
        self.use_dataloader = use_dataloader
        # set init data to None
        feat = None
        gt_start = None
        gt_end = None
        gt_action = None
        fileid = None

        feat = fluid.data(name='feat', shape=feat_shape, dtype='float32')

        feed_list = []
        feed_list.append(feat)
        if (self.mode == 'train') or (self.mode == 'valid'):
            gt_start = fluid.data(
                name='gt_start', shape=gt_start_shape, dtype='float32')
            gt_end = fluid.data(
                name='gt_end', shape=gt_end_shape, dtype='float32')
            gt_action = fluid.data(
                name='gt_action', shape=gt_action_shape, dtype='float32')
            feed_list.append(gt_start)
            feed_list.append(gt_end)
            feed_list.append(gt_action)

        elif self.mode == 'test':
            gt_start = fluid.data(
                name='gt_start', shape=gt_start_shape, dtype='float32')
            gt_end = fluid.data(
                name='gt_end', shape=gt_end_shape, dtype='float32')
            gt_action = fluid.data(
                name='gt_action', shape=gt_action_shape, dtype='float32')
            feed_list.append(gt_start)
            feed_list.append(gt_end)
            feed_list.append(gt_action)
            fileid = fluid.data(
                name='fileid', shape=fileid_shape, dtype='int64')
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
                feed_list=feed_list, capacity=8, iterable=True)

        self.feat_input = [feat]
        self.gt_start = gt_start
        self.gt_end = gt_end
        self.gt_action = gt_action
        self.fileid = fileid

    def create_model_args(self):
        cfg = {}
        cfg['tscale'] = self.tscale
        cfg['feat_dim'] = self.feat_dim
        cfg['hidden_dim'] = self.hidden_dim
        return cfg

    def build_model(self):
        cfg = self.create_model_args()
        self.videomodel = bsn_net.BsnTemNet(cfg=cfg)
        preds = self.videomodel.net(input=self.feat_input[0])
        self.network_outputs = [preds]

    def optimizer(self):
        bd = [self.lr_decay_iter]
        base_lr = self.base_learning_rate
        lr_decay = self.learning_rate_decay
        lr = [base_lr, base_lr * lr_decay]
        l2_weight_decay = self.l2_weight_decay
        optimizer = fluid.optimizer.Adam(
            fluid.layers.piecewise_decay(
                boundaries=bd, values=lr),
            regularization=fluid.regularizer.L2DecayRegularizer(
                regularization_coeff=l2_weight_decay))
        return optimizer

    def loss(self):
        assert self.mode != 'infer', "invalid loss calculationg in infer mode"
        self.loss_ = self.videomodel.loss_func(
            self.network_outputs[0], self.gt_start, self.gt_end, self.gt_action)
        return self.loss_

    def outputs(self):
        return self.network_outputs

    def feeds(self):
        if (self.mode == 'train') or (self.mode == 'valid'):
            return self.feat_input + [
                self.gt_start, self.gt_end, self.gt_action
            ]
        elif self.mode == 'test':
            return self.feat_input + [
                self.gt_start, self.gt_end, self.gt_action, self.fileid
            ]
        elif self.mode == 'infer':
            return self.feat_input
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
            'BsnTem.pdparams',
            'https://paddlemodels.bj.bcebos.com/video_detection/BsnTem.pdparams'
        )


class BsnPem(ModelBase):
    """BsnPem model"""

    def __init__(self, name, cfg, mode='train'):
        super(BsnPem, self).__init__(name, cfg, mode=mode)
        self.mode = mode
        self.get_config()

    def get_config(self):
        self.feat_dim = self.get_config_from_sec('MODEL', 'feat_dim')
        self.hidden_dim = self.get_config_from_sec('MODEL', 'hidden_dim')

        self.batch_size = self.get_config_from_sec(self.mode, 'batch_size')
        self.num_gpus = self.get_config_from_sec(self.mode, 'num_gpus')
        self.top_K = self.get_config_from_sec(self.mode, 'top_K')

        self.num_epochs = self.get_config_from_sec('train', 'epoch')
        self.base_learning_rate = self.get_config_from_sec('train',
                                                           'learning_rate')
        self.learning_rate_decay = self.get_config_from_sec(
            'train', 'learning_rate_decay')
        self.l2_weight_decay = self.get_config_from_sec('train',
                                                        'l2_weight_decay')
        self.lr_decay_iter = self.get_config_from_sec('train', 'lr_decay_iter')

    def build_input(self, use_dataloader=True):
        feat_shape = [None, self.top_K, self.feat_dim]
        gt_iou_shape = [None, self.top_K, 1]
        props_info_shape = [None, self.top_K, 4]
        fileid_shape = [None, 1]
        self.use_dataloader = use_dataloader
        # set init data to None
        feat = None
        gt_iou = None
        props_info = None
        fileid = None

        feat = fluid.data(name='feat', shape=feat_shape, dtype='float32')

        feed_list = []
        feed_list.append(feat)
        if (self.mode == 'train') or (self.mode == 'valid'):
            gt_iou = fluid.data(
                name='gt_iou', shape=gt_iou_shape, dtype='float32')
            feed_list.append(gt_iou)

        elif self.mode == 'test':
            gt_iou = fluid.data(
                name='gt_iou', shape=gt_iou_shape, dtype='float32')
            props_info = fluid.data(
                name='props_info', shape=props_info_shape, dtype='float32')
            feed_list.append(gt_iou)
            feed_list.append(props_info)
            fileid = fluid.data(
                name='fileid', shape=fileid_shape, dtype='int64')
            feed_list.append(fileid)
        elif self.mode == 'infer':
            pass
        else:
            raise NotImplementedError('mode {} not implemented'.format(
                self.mode))

        if use_dataloader:
            assert self.mode != 'infer', \
                        'dataloader is not recommendated when infer, please set use_dataloader to be false.'
            self.dataloader = fluid.io.DataLoader.from_generator(
                feed_list=feed_list, capacity=4, iterable=True)

        self.feat_input = [feat]
        self.gt_iou = gt_iou
        self.props_info = props_info
        self.fileid = fileid

    def create_model_args(self):
        cfg = {}
        cfg['feat_dim'] = self.feat_dim
        cfg['hidden_dim'] = self.hidden_dim
        cfg['batch_size'] = self.batch_size
        cfg['top_K'] = self.top_K
        cfg["num_gpus"] = self.num_gpus
        return cfg

    def build_model(self):
        cfg = self.create_model_args()
        self.videomodel = bsn_net.BsnPemNet(cfg=cfg)
        preds = self.videomodel.net(input=self.feat_input[0])
        self.network_outputs = [preds]

    def optimizer(self):
        bd = [self.lr_decay_iter]
        base_lr = self.base_learning_rate
        lr_decay = self.learning_rate_decay
        lr = [base_lr, base_lr * lr_decay]
        l2_weight_decay = self.l2_weight_decay
        optimizer = fluid.optimizer.Adam(
            fluid.layers.piecewise_decay(
                boundaries=bd, values=lr),
            regularization=fluid.regularizer.L2DecayRegularizer(
                regularization_coeff=l2_weight_decay))
        return optimizer

    def loss(self):
        assert self.mode != 'infer', "invalid loss calculationg in infer mode"
        self.loss_ = self.videomodel.loss_func(self.network_outputs[0],
                                               self.gt_iou)
        return self.loss_

    def outputs(self):
        return self.network_outputs

    def feeds(self):
        if (self.mode == 'train') or (self.mode == 'valid'):
            return self.feat_input + [self.gt_iou]
        elif self.mode == 'test':
            return self.feat_input + [self.gt_iou, self.props_info, self.fileid]
        elif self.mode == 'infer':
            return self.feat_input
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
                         [self.props_info, self.fileid]
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
            'BsnPem.pdparams',
            'https://paddlemodels.bj.bcebos.com/video_detection/BsnPem.pdparams'
        )
