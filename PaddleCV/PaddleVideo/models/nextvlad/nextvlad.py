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

import numpy as np

import paddle.fluid as fluid
from paddle.fluid import ParamAttr

from ..model import ModelBase
from .clf_model import LogisticModel
from . import nextvlad_model

__all__ = ["NEXTVLAD"]


class NEXTVLAD(ModelBase):
    def __init__(self, name, cfg, mode='train'):
        super(NEXTVLAD, self).__init__(name, cfg, mode=mode)
        self.get_config()

    def get_config(self):
        # model params
        self.num_classes = self.get_config_from_sec('model', 'num_classes')
        self.video_feature_size = self.get_config_from_sec('model',
                                                           'video_feature_size')
        self.audio_feature_size = self.get_config_from_sec('model',
                                                           'audio_feature_size')
        self.cluster_size = self.get_config_from_sec('model', 'cluster_size')
        self.hidden_size = self.get_config_from_sec('model', 'hidden_size')
        self.groups = self.get_config_from_sec('model', 'groups')
        self.expansion = self.get_config_from_sec('model', 'expansion')
        self.drop_rate = self.get_config_from_sec('model', 'drop_rate')
        self.gating_reduction = self.get_config_from_sec('model',
                                                         'gating_reduction')
        self.eigen_file = self.get_config_from_sec('model', 'eigen_file')
        # training params
        self.base_learning_rate = self.get_config_from_sec('train',
                                                           'learning_rate')
        self.lr_boundary_examples = self.get_config_from_sec(
            'train', 'lr_boundary_examples')
        self.max_iter = self.get_config_from_sec('train', 'max_iter')
        self.learning_rate_decay = self.get_config_from_sec(
            'train', 'learning_rate_decay')
        self.l2_penalty = self.get_config_from_sec('train', 'l2_penalty')
        self.gradient_clip_norm = self.get_config_from_sec('train',
                                                           'gradient_clip_norm')
        self.use_gpu = self.get_config_from_sec('train', 'use_gpu')
        self.num_gpus = self.get_config_from_sec('train', 'num_gpus')

        # other params
        self.batch_size = self.get_config_from_sec(self.mode, 'batch_size')

    def build_input(self, use_dataloader=True):
        rgb_shape = [None, self.video_feature_size]
        audio_shape = [None, self.audio_feature_size]
        label_shape = [None, self.num_classes]

        rgb = fluid.data(
            name='train_rgb' if self.is_training else 'test_rgb',
            shape=rgb_shape,
            dtype='uint8',
            lod_level=1)
        audio = fluid.data(
            name='train_audio' if self.is_training else 'test_audio',
            shape=audio_shape,
            dtype='uint8',
            lod_level=1)
        if self.mode == 'infer':
            label = None
        else:
            label = fluid.data(
                name='train_label' if self.is_training else 'test_label',
                shape=label_shape,
                dtype='float32')

        if use_dataloader:
            assert self.mode != 'infer', \
                    'dataloader is not recommendated when infer, please set use_dataloader to be false.'
            self.dataloader = fluid.io.DataLoader.from_generator(
                feed_list=[rgb, audio, label], capacity=8, iterable=True)
        self.feature_input = [rgb, audio]
        self.label_input = label

    def create_model_args(self):
        model_args = {}
        model_args['class_dim'] = self.num_classes
        model_args['cluster_size'] = self.cluster_size
        model_args['hidden_size'] = self.hidden_size
        model_args['groups'] = self.groups
        model_args['expansion'] = self.expansion
        model_args['drop_rate'] = self.drop_rate
        model_args['gating_reduction'] = self.gating_reduction
        model_args['l2_penalty'] = self.l2_penalty
        return model_args

    def build_model(self):
        model_args = self.create_model_args()
        videomodel = nextvlad_model.NeXtVLADModel()
        rgb = self.feature_input[0]
        audio = self.feature_input[1]

        # move data processing from data reader to fluid to process on gpu
        rgb = fluid.layers.cast(rgb, 'float32')
        audio = fluid.layers.cast(audio, 'float32')
        bias = -2.
        scale = 4. / 255
        offset = 4. / 512

        rgb = fluid.layers.scale(rgb, scale=scale, bias=bias)
        audio = fluid.layers.scale(audio, scale=scale, bias=bias + offset)

        eigen_value = np.sqrt(np.load(self.eigen_file)[:1024, 0])
        eigen_value = (eigen_value + 1e-4).astype(np.float32)
        eigen_param = fluid.layers.create_parameter(
            shape=eigen_value.shape,
            dtype='float32',
            attr=fluid.ParamAttr(
                name='eigen_param', trainable=False),
            default_initializer=fluid.initializer.NumpyArrayInitializer(
                eigen_value))

        rgb = fluid.layers.elementwise_mul(rgb, eigen_param)
        rgb.stop_gradient = True
        audio.stop_gradient = True

        out = videomodel.create_model(
            rgb, audio, is_training=(self.mode == 'train'), **model_args)
        self.logits = out['logits']
        self.predictions = out['predictions']
        self.network_outputs = [out['predictions']]

    def optimizer(self):
        assert self.mode == 'train', "optimizer only can be get in train mode"
        im_per_batch = self.batch_size
        lr_bounds, lr_values = get_learning_rate_decay_list(
            self.base_learning_rate, self.learning_rate_decay, self.max_iter,
            self.lr_boundary_examples, im_per_batch)
        return fluid.optimizer.AdamOptimizer(
            learning_rate=fluid.layers.piecewise_decay(
                boundaries=lr_bounds, values=lr_values))

    def loss(self):
        assert self.mode != 'infer', "invalid loss calculationg in infer mode"
        cost = fluid.layers.sigmoid_cross_entropy_with_logits(
            x=self.logits, label=self.label_input)
        cost = fluid.layers.reduce_sum(cost, dim=-1)
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
            fetch_list = [losses, self.predictions, self.label_input]
        elif self.mode == 'test':
            losses = self.loss()
            fetch_list = [losses, self.predictions, self.label_input]
        elif self.mode == 'infer':
            fetch_list = [self.predictions]
        else:
            raise NotImplementedError('mode {} not implemented'.format(
                self.mode))

        return fetch_list

    def weights_info(self):
        return (
            'NEXTVLAD.pdparams',
            'https://paddlemodels.bj.bcebos.com/video_classification/NEXTVLAD.pdparams'
        )


def get_learning_rate_decay_list(base_learning_rate, decay, max_iter,
                                 decay_examples, total_batch_size):
    decay_step = decay_examples // total_batch_size
    lr_bounds = []
    lr_values = [base_learning_rate]
    i = 1
    while True:
        if i * decay_step >= max_iter:
            break
        lr_bounds.append(i * decay_step)
        lr_values.append(base_learning_rate * (decay**i))
        i += 1
    return lr_bounds, lr_values
