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
from . import tall_net

import logging
logger = logging.getLogger(__name__)

__all__ = ["TALL"]


class TALL(ModelBase):
    """TALL model"""

    def __init__(self, name, cfg, mode='train'):
        super(TALL, self).__init__(name, cfg, mode=mode)
        self.get_config()

    def get_config(self):
        self.visual_feature_dim = self.get_config_from_sec('MODEL',
                                                           'visual_feature_dim')
        self.sentence_embedding_size = self.get_config_from_sec(
            'MODEL', 'sentence_embedding_size')
        self.semantic_size = self.get_config_from_sec('MODEL', 'semantic_size')
        self.hidden_size = self.get_config_from_sec('MODEL', 'hidden_size')
        self.output_size = self.get_config_from_sec('MODEL', 'output_size')
        self.batch_size = self.get_config_from_sec(self.mode, 'batch_size')

        self.off_size = self.get_config_from_sec('train',
                                                 'off_size')  # in train of yaml
        self.clip_norm = self.get_config_from_sec('train', 'clip_norm')
        self.learning_rate = self.get_config_from_sec('train', 'learning_rate')

    def build_input(self, use_dataloader=True):
        visual_shape = self.visual_feature_dim
        sentence_shape = self.sentence_embedding_size
        offset_shape = self.off_size

        # set init data to None
        images = None
        sentences = None
        offsets = None

        self.use_dataloader = use_dataloader

        images = fluid.data(
            name='train_visual', shape=[None, visual_shape], dtype='float32')

        sentences = fluid.data(
            name='train_sentences',
            shape=[None, sentence_shape],
            dtype='float32')

        feed_list = []
        feed_list.append(images)
        feed_list.append(sentences)
        if (self.mode == 'train') or (self.mode == 'valid'):
            offsets = fluid.data(
                name='train_offsets',
                shape=[None, offset_shape],
                dtype='float32')

            feed_list.append(offsets)
        elif (self.mode == 'test') or (self.mode == 'infer'):
            # input images and sentences
            pass
        else:
            raise NotImplementedError('mode {} not implemented'.format(
                self.mode))

        if use_dataloader:
            assert self.mode != 'infer', \
                        'dataloader is not recommendated when infer, please set use_dataloader to be false.'
            self.dataloader = fluid.io.DataLoader.from_generator(
                feed_list=feed_list, capacity=16, iterable=True)

        self.images = [images]
        self.sentences = sentences
        self.offsets = offsets

    def create_model_args(self):
        cfg = {}

        cfg['semantic_size'] = self.semantic_size
        cfg['sentence_embedding_size'] = self.sentence_embedding_size
        cfg['hidden_size'] = self.hidden_size
        cfg['output_size'] = self.output_size
        cfg['batch_size'] = self.batch_size
        return cfg

    def build_model(self):
        cfg = self.create_model_args()
        self.videomodel = tall_net.TALLNET(
            semantic_size=cfg['semantic_size'],
            sentence_embedding_size=cfg['sentence_embedding_size'],
            hidden_size=cfg['hidden_size'],
            output_size=cfg['output_size'],
            batch_size=cfg['batch_size'],
            mode=self.mode)
        outs = self.videomodel.net(images=self.images[0],
                                   sentences=self.sentences)
        self.network_outputs = [outs]

    def optimizer(self):
        clip_norm = self.clip_norm

        fluid.clip.set_gradient_clip(
            clip=fluid.clip.GradientClipByGlobalNorm(clip_norm=clip_norm))

        optimizer = fluid.optimizer.Adam(learning_rate=self.learning_rate)

        return optimizer

    def loss(self):
        assert self.mode != 'infer', "invalid loss calculationg in infer mode"
        self.loss_ = self.videomodel.loss(self.network_outputs[0], self.offsets)
        return self.loss_

    def outputs(self):
        preds = self.network_outputs[0]
        return [preds]

    def feeds(self):
        if (self.mode == 'train') or (self.mode == 'valid'):
            return self.images + [self.sentences, self.offsets]
        elif self.mode == 'test' or (self.mode == 'infer'):
            return self.images + [self.sentences]
        else:
            raise NotImplementedError('mode {} not implemented'.format(
                self.mode))

    def fetches(self):
        if (self.mode == 'train') or (self.mode == 'valid'):
            losses = self.loss()
            fetch_list = [item for item in losses]
        elif (self.mode == 'test') or (self.mode == 'infer'):
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
            'TALL.pdparams',
            'https://paddlemodels.bj.bcebos.com/video_grounding/TALL.pdparams')
