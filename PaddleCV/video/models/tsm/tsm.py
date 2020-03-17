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

import paddle.fluid as fluid
from paddle.fluid import ParamAttr

from ..model import ModelBase
from .tsm_res_model import TSM_ResNet

import logging
logger = logging.getLogger(__name__)

__all__ = ["TSM"]


class TSM(ModelBase):
    def __init__(self, name, cfg, mode='train'):
        super(TSM, self).__init__(name, cfg, mode=mode)
        self.get_config()

    def get_config(self):
        self.num_classes = self.get_config_from_sec('model', 'num_classes')
        self.seg_num = self.get_config_from_sec('model', 'seg_num')
        self.seglen = self.get_config_from_sec('model', 'seglen')
        self.image_mean = self.get_config_from_sec('model', 'image_mean')
        self.image_std = self.get_config_from_sec('model', 'image_std')
        self.num_layers = self.get_config_from_sec('model', 'num_layers')

        self.num_epochs = self.get_config_from_sec('train', 'epoch')
        self.total_videos = self.get_config_from_sec('train', 'total_videos')
        self.base_learning_rate = self.get_config_from_sec('train',
                                                           'learning_rate')
        self.learning_rate_decay = self.get_config_from_sec(
            'train', 'learning_rate_decay')
        self.decay_epochs = self.get_config_from_sec('train', 'decay_epochs')
        self.l2_weight_decay = self.get_config_from_sec('train',
                                                        'l2_weight_decay')
        self.momentum = self.get_config_from_sec('train', 'momentum')

        self.target_size = self.get_config_from_sec(self.mode, 'target_size')
        self.batch_size = self.get_config_from_sec(self.mode, 'batch_size')

    def build_input(self, use_dataloader=True):
        image_shape = [3, self.target_size, self.target_size]
        image_shape[0] = image_shape[0] * self.seglen
        image_shape = [None, self.seg_num] + image_shape
        self.use_dataloader = use_dataloader

        image = fluid.data(name='image', shape=image_shape, dtype='float32')
        if self.mode != 'infer':
            label = fluid.data(name='label', shape=[None, 1], dtype='int64')
        else:
            label = None

        if use_dataloader:
            assert self.mode != 'infer', \
                        'dataloader is not recommendated when infer, please set use_dataloader to be false.'
            self.dataloader = fluid.io.DataLoader.from_generator(
                feed_list=[image, label], capacity=4, iterable=True)

        self.feature_input = [image]
        self.label_input = label

    def build_model(self):
        videomodel = TSM_ResNet(
            layers=self.num_layers,
            seg_num=self.seg_num,
            is_training=self.is_training)
        out = videomodel.net(input=self.feature_input[0],
                             class_dim=self.num_classes)
        self.network_outputs = [out]

    def optimizer(self):
        assert self.mode == 'train', "optimizer only can be get in train mode"
        total_videos = self.total_videos
        step = int(total_videos / self.batch_size + 1)
        bd = [e * step for e in self.decay_epochs]
        base_lr = self.base_learning_rate
        lr_decay = self.learning_rate_decay
        lr = [base_lr, base_lr * lr_decay, base_lr * lr_decay * lr_decay]
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
        cost = fluid.layers.cross_entropy(input=self.network_outputs[0], \
                           label=self.label_input, ignore_index=-1)
        self.loss_ = fluid.layers.mean(x=cost)
        return self.loss_

    def outputs(self):
        return self.network_outputs

    def feeds(self):
        return self.feature_input if self.mode == 'infer' else self.feature_input + [
            self.label_input
        ]

    def fetches(self):
        if self.mode == 'train' or self.mode == 'valid':
            losses = self.loss()
            fetch_list = [losses, self.network_outputs[0], self.label_input]
        elif self.mode == 'test':
            losses = self.loss()
            fetch_list = [losses, self.network_outputs[0], self.label_input]
        elif self.mode == 'infer':
            fetch_list = self.network_outputs
        else:
            raise NotImplementedError('mode {} not implemented'.format(
                self.mode))

        return fetch_list

    def pretrain_info(self):
        return (
            'ResNet50_pretrained',
            'https://paddlemodels.bj.bcebos.com/video_classification/ResNet50_pretrained.tar.gz'
        )

    def weights_info(self):
        return (
            'TSM.pdparams',
            'https://paddlemodels.bj.bcebos.com/video_classification/TSM.pdparams'
        )

    def load_pretrain_params(self, exe, pretrain, prog, place):
        def is_parameter(var):
            return isinstance(var, fluid.framework.Parameter) and (
                not ("fc_0" in var.name))

        logger.info("Load pretrain weights from {}, exclude fc layer.".format(
            pretrain))

        state_dict = fluid.load_program_state(pretrain)
        dict_keys = list(state_dict.keys())
        for name in dict_keys:
            if "fc_0" in name:
                del state_dict[name]
                print('Delete {} from pretrained parameters. Do not load it'.
                      format(name))
        fluid.set_program_state(prog, state_dict)
