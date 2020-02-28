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
from . import ets_net

import logging
logger = logging.getLogger(__name__)

__all__ = ["ETS"]


class ETS(ModelBase):
    """ETS model"""

    def __init__(self, name, cfg, mode='train'):
        super(ETS, self).__init__(name, cfg, mode=mode)
        self.get_config()

    def get_config(self):
        self.feat_size = self.get_config_from_sec('MODEL', 'feat_size')
        self.fc_dim = self.get_config_from_sec('MODEL', 'fc_dim')
        self.gru_hidden_dim = self.get_config_from_sec('MODEL',
                                                       'gru_hidden_dim')
        self.decoder_size = self.get_config_from_sec('MODEL', 'decoder_size')
        self.word_emb_dim = self.get_config_from_sec('MODEL', 'word_emb_dim')
        self.dict_file = self.get_config_from_sec('MODEL', 'dict_file')
        self.max_length = self.get_config_from_sec('MODEL', 'max_length')
        self.beam_size = self.get_config_from_sec('MODEL', 'beam_size')

        self.num_epochs = self.get_config_from_sec('train', 'epoch')
        self.l2_weight_decay = self.get_config_from_sec('train',
                                                        'l2_weight_decay')
        self.clip_norm = self.get_config_from_sec('train', 'clip_norm')

    def build_input(self, use_dataloader=True):
        feat_shape = [None, self.feat_size]
        word_shape = [None, 1]
        word_next_shape = [None, 1]

        # set init data to None
        py_reader = None
        feat = None
        word = None
        word_next = None
        init_ids = None
        init_scores = None

        self.use_dataloader = use_dataloader
        feat = fluid.data(
            name='feat', shape=feat_shape, dtype='float32', lod_level=1)

        feed_list = []
        feed_list.append(feat)
        if (self.mode == 'train') or (self.mode == 'valid'):
            word = fluid.data(
                name='word', shape=word_shape, dtype='int64', lod_level=1)
            word_next = fluid.data(
                name='word_next',
                shape=word_next_shape,
                dtype='int64',
                lod_level=1)
            feed_list.append(word)
            feed_list.append(word_next)
        elif (self.mode == 'test') or (self.mode == 'infer'):
            init_ids = fluid.data(
                name="init_ids", shape=[None, 1], dtype="int64", lod_level=2)
            init_scores = fluid.data(
                name="init_scores",
                shape=[None, 1],
                dtype="float32",
                lod_level=2)
        else:
            raise NotImplementedError('mode {} not implemented'.format(
                self.mode))

        if use_dataloader:
            assert self.mode != 'infer', \
                        'dataloader is not recommendated when infer, please set use_dataloader to be false.'
            self.dataloader = fluid.io.DataLoader.from_generator(
                feed_list=feed_list, capacity=16, iterable=True)

        self.feature_input = [feat]
        self.word = word
        self.word_next = word_next
        self.init_ids = init_ids
        self.init_scores = init_scores

    def create_model_args(self):
        cfg = {}
        cfg['feat_size'] = self.feat_size
        cfg['fc_dim'] = self.fc_dim
        cfg['gru_hidden_dim'] = self.gru_hidden_dim
        cfg['decoder_size'] = self.decoder_size
        cfg['word_emb_dim'] = self.word_emb_dim
        word_dict = dict()
        with open(self.dict_file, 'r') as f:
            for i, line in enumerate(f):
                word_dict[line.strip().split()[0]] = i
        dict_size = len(word_dict)
        cfg['dict_size'] = dict_size
        cfg['max_length'] = self.max_length
        cfg['beam_size'] = self.beam_size

        return cfg

    def build_model(self):
        cfg = self.create_model_args()
        self.videomodel = ets_net.ETSNET(
            feat_size=cfg['feat_size'],
            fc_dim=cfg['fc_dim'],
            gru_hidden_dim=cfg['gru_hidden_dim'],
            decoder_size=cfg['decoder_size'],
            word_emb_dim=cfg['word_emb_dim'],
            dict_size=cfg['dict_size'],
            max_length=cfg['max_length'],
            beam_size=cfg['beam_size'],
            mode=self.mode)
        if (self.mode == 'train') or (self.mode == 'valid'):
            prob = self.videomodel.net(self.feature_input[0], self.word)
            self.network_outputs = [prob]
        elif (self.mode == 'test') or (self.mode == 'infer'):
            translation_ids, translation_scores = self.videomodel.net(
                self.feature_input[0], self.init_ids, self.init_scores)
            self.network_outputs = [translation_ids, translation_scores]

    def optimizer(self):
        l2_weight_decay = self.l2_weight_decay
        clip_norm = self.clip_norm

        fluid.clip.set_gradient_clip(
            clip=fluid.clip.GradientClipByGlobalNorm(clip_norm=clip_norm))
        lr_decay = fluid.layers.learning_rate_scheduler.noam_decay(
            self.gru_hidden_dim, 1000)
        optimizer = fluid.optimizer.Adam(
            learning_rate=lr_decay,
            regularization=fluid.regularizer.L2DecayRegularizer(
                regularization_coeff=l2_weight_decay))

        return optimizer

    def loss(self):
        assert self.mode != 'infer', "invalid loss calculationg in infer mode"
        self.loss_ = self.videomodel.loss(self.network_outputs[0],
                                          self.word_next)
        return self.loss_

    def outputs(self):
        return self.network_outputs

    def feeds(self):
        if (self.mode == 'train') or (self.mode == 'valid'):
            return self.feature_input + [self.word, self.word_next]
        elif (self.mode == 'test') or (self.mode == 'infer'):
            return self.feature_input + [self.init_ids, self.init_scores]
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
        return ('ETS.pdparams',
                'https://paddlemodels.bj.bcebos.com/video_caption/ETS.pdparams')
