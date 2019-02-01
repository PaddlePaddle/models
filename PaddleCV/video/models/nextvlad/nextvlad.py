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

    def build_input(self, use_pyreader=True):
        rgb_shape = [self.video_feature_size]
        audio_shape = [self.audio_feature_size]
        label_shape = [self.num_classes]
        if use_pyreader:
            assert self.mode != 'infer', \
                      'pyreader is not recommendated when infer, please set use_pyreader to be false.'
            py_reader = fluid.layers.py_reader(
                capacity=100,
                shapes=[[-1] + rgb_shape, [-1] + audio_shape,
                        [-1] + label_shape],
                lod_levels=[1, 1, 0],
                dtypes=['float32', 'float32', 'float32'],
                name='train_py_reader'
                if self.is_training else 'test_py_reader',
                use_double_buffer=True)
            rgb, audio, label = fluid.layers.read_file(py_reader)
            self.py_reader = py_reader
        else:
            rgb = fluid.layers.data(
                name='train_rgb' if self.is_training else 'test_rgb',
                shape=rgb_shape,
                dtype='float32',
                lod_level=1)
            audio = fluid.layers.data(
                name='train_audio' if self.is_training else 'test_audio',
                shape=audio_shape,
                dtype='float32',
                lod_level=1)
            if self.mode == 'infer':
                label = None
            else:
                label = fluid.layers.data(
                    name='train_label' if self.is_training else 'test_label',
                    shape=label_shape,
                    dtype='float32')
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
    
    def weights_info(self):
        return ('nextvlad_youtube8m', 
                'https://paddlemodels.bj.bcebos.com/video_classification/nextvlad_youtube8m.tar.gz')


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
