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

import os
import numpy as np
import paddle.fluid as fluid

from ..model import ModelBase
from . import resnet_video
from .nonlocal_utils import load_pretrain_params_from_file, load_weights_params_from_file

import logging
logger = logging.getLogger(__name__)

__all__ = ["NonLocal"]

# To add new models, import them, add them to this map and models/TARGETS


class NonLocal(ModelBase):
    def __init__(self, name, cfg, mode='train'):
        super(NonLocal, self).__init__(name, cfg, mode=mode)
        self.get_config()

    def get_config(self):
        # video_length
        self.video_length = self.get_config_from_sec(self.mode, 'video_length')
        # crop size
        self.crop_size = self.get_config_from_sec(self.mode, 'crop_size')

    def build_input(self, use_dataloader=True):
        input_shape = [
            None, 3, self.video_length, self.crop_size, self.crop_size
        ]
        label_shape = [None, 1]

        data = fluid.data(
            name='train_data' if self.is_training else 'test_data',
            shape=input_shape,
            dtype='float32')
        if self.mode != 'infer':
            label = fluid.data(
                name='train_label' if self.is_training else 'test_label',
                shape=label_shape,
                dtype='int64')
        else:
            label = None

        if use_dataloader:
            assert self.mode != 'infer', \
                        'dataloader is not recommendated when infer, please set use_dataloader to be false.'
            self.dataloader = fluid.io.DataLoader.from_generator(
                feed_list=[data, label], capacity=4, iterable=True)

        self.feature_input = [data]
        self.label_input = label

    def create_model_args(self):
        return None

    def build_model(self):
        pred, loss = resnet_video.create_model(
            data=self.feature_input[0],
            label=self.label_input,
            cfg=self.cfg,
            is_training=self.is_training,
            mode=self.mode)
        if loss is not None:
            loss = fluid.layers.mean(loss)
        self.network_outputs = [pred]
        self.loss_ = loss

    def optimizer(self):
        base_lr = self.get_config_from_sec('TRAIN', 'learning_rate')
        lr_decay = self.get_config_from_sec('TRAIN', 'learning_rate_decay')
        step_sizes = self.get_config_from_sec('TRAIN', 'step_sizes')
        lr_bounds, lr_values = get_learning_rate_decay_list(base_lr, lr_decay,
                                                            step_sizes)
        learning_rate = fluid.layers.piecewise_decay(
            boundaries=lr_bounds, values=lr_values)

        momentum = self.get_config_from_sec('TRAIN', 'momentum')
        use_nesterov = self.get_config_from_sec('TRAIN', 'nesterov')
        l2_weight_decay = self.get_config_from_sec('TRAIN', 'weight_decay')
        logger.info(
            'Build up optimizer, \ntype: {}, \nmomentum: {}, \nnesterov: {}, \
                                         \nregularization: L2 {}, \nlr_values: {}, lr_bounds: {}'
            .format('Momentum', momentum, use_nesterov, l2_weight_decay,
                    lr_values, lr_bounds))
        optimizer = fluid.optimizer.Momentum(
            learning_rate=learning_rate,
            momentum=momentum,
            use_nesterov=use_nesterov,
            regularization=fluid.regularizer.L2Decay(l2_weight_decay))
        return optimizer

    def loss(self):
        return self.loss_

    def outputs(self):
        return self.network_outputs

    def feeds(self):
        return self.feature_input if self.mode == 'infer' else \
                     self.feature_input + [self.label_input]

    def fetches(self):
        if self.mode == 'train' or self.mode == 'valid':
            losses = self.loss()
            fetch_list = [losses, self.network_outputs[0], self.label_input]
        elif self.mode == 'test':
            fetch_list = [self.network_outputs[0], self.label_input]
        elif self.mode == 'infer':
            fetch_list = self.network_outputs
        else:
            raise NotImplementedError('mode {} not implemented'.format(
                self.mode))

        return fetch_list

    def pretrain_info(self):
        return (
            'Nonlocal_ResNet50_pretrained',
            'https://paddlemodels.bj.bcebos.com/video_classification/Nonlocal_ResNet50_pretrained.tar.gz'
        )

    def weights_info(self):
        return (
            'NONLOCAL.pdparams',
            'https://paddlemodels.bj.bcebos.com/video_classification/NONLOCAL.pdparams'
        )

    def load_pretrain_params(self, exe, pretrain, prog, place):
        load_pretrain_params_from_file(exe, prog, pretrain, place)

    def load_test_weights(self, exe, weights, prog, place):
        load_weights_params_from_file(exe, prog, weights, place)


def get_learning_rate_decay_list(base_learning_rate, lr_decay, step_lists):
    lr_bounds = []
    lr_values = [base_learning_rate * 1]
    cur_step = 0
    for i in range(len(step_lists)):
        cur_step += step_lists[i]
        lr_bounds.append(cur_step)
        decay_rate = lr_decay**(i + 1)
        lr_values.append(base_learning_rate * decay_rate)

    return lr_bounds, lr_values
