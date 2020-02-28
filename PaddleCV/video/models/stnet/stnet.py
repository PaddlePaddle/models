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

from ..model import ModelBase
from .stnet_res_model import StNet_ResNet

import logging
logger = logging.getLogger(__name__)

__all__ = ["STNET"]


class STNET(ModelBase):
    def __init__(self, name, cfg, mode='train'):
        super(STNET, self).__init__(name, cfg, mode=mode)
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
        self.l2_weight_decay = self.get_config_from_sec('train',
                                                        'l2_weight_decay')
        self.momentum = self.get_config_from_sec('train', 'momentum')

        self.seg_num = self.get_config_from_sec(self.mode, 'seg_num',
                                                self.seg_num)
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

    def create_model_args(self):
        cfg = {}
        cfg['layers'] = self.num_layers
        cfg['class_dim'] = self.num_classes
        cfg['seg_num'] = self.seg_num
        cfg['seglen'] = self.seglen
        return cfg

    def build_model(self):
        cfg = self.create_model_args()
        videomodel = StNet_ResNet(layers = cfg['layers'], seg_num = cfg['seg_num'], \
                                  seglen = cfg['seglen'], is_training = (self.mode == 'train'))
        out = videomodel.net(input=self.feature_input[0],
                             class_dim=cfg['class_dim'])
        self.network_outputs = [out]

    def optimizer(self):
        epoch_points = [self.num_epochs / 3, self.num_epochs * 2 / 3]
        total_videos = self.total_videos
        step = int(total_videos / self.batch_size + 1)
        bd = [e * step for e in epoch_points]
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
            'STNET.pdparams',
            'https://paddlemodels.bj.bcebos.com/video_classification/STNET.pdparams'
        )

    def load_pretrain_params(self, exe, pretrain, prog, place):
        """
        The pretrained params are ResNet50 pretrained on ImageNet.
        However, conv1_weights' shape of StNet is not the same as that in ResNet50 because the input are super-image
        concatanated by a series of images. So it is recommendated to treat conv1_weights specifically.
        The process is as following:
          1, load params from pretrain
          2, get the value of conv1_weights in the state_dict and transform it
          3, set the transformed value to conv1_weights in prog
        """

        logger.info(
            "Load pretrain weights from {}, exclude fc, batch_norm, xception, conv3d layers.".
            format(pretrain))

        state_dict = fluid.load_program_state(pretrain)
        dict_keys = list(state_dict.keys())
        for name in dict_keys:
            if ("batch_norm" in name) or ("fc_0" in name) or ("batch_norm" in name) \
                     or ("xception" in name) or ("conv3d" in name):
                del state_dict[name]
                logger.info(
                    'Delete {} from pretrained parameters. Do not load it'.
                    format(name))
        conv1_weights = state_dict["conv1_weights"]
        conv1_weights = np.mean(
            conv1_weights, axis=1, keepdims=True) / self.seglen
        conv1_weights = np.repeat(conv1_weights, 3 * self.seglen, axis=1)
        logger.info(
            'conv1_weights is transformed from [Cout, 3, Kh, Kw] into [Cout, 3*seglen, Kh, Kw]'
        )
        state_dict["conv1_weights"] = conv1_weights
        fluid.set_program_state(prog, state_dict)
