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

        self.seg_num = self.get_config_from_sec(self.mode, 'seg_num', self.seg_num)
        self.target_size = self.get_config_from_sec(self.mode, 'target_size')
        self.batch_size = self.get_config_from_sec(self.mode, 'batch_size')

    def build_input(self, use_pyreader=True):
        image_shape = [3, self.target_size, self.target_size]
        image_shape[0] = image_shape[0] * self.seglen
        image_shape = [self.seg_num] + image_shape
        self.use_pyreader = use_pyreader
        if use_pyreader:
            assert self.mode != 'infer', \
                        'pyreader is not recommendated when infer, please set use_pyreader to be false.'
            py_reader = fluid.layers.py_reader(
                capacity=100,
                shapes=[[-1] + image_shape, [-1] + [1]],
                dtypes=['float32', 'int64'],
                name='train_py_reader'
                if self.is_training else 'test_py_reader',
                use_double_buffer=True)
            image, label = fluid.layers.read_file(py_reader)
            self.py_reader = py_reader
        else:
            image = fluid.layers.data(
                name='image', shape=image_shape, dtype='float32')
            if self.mode != 'infer':
                label = fluid.layers.data(
                    name='label', shape=[1], dtype='int64')
            else:
                label = None
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

    def pretrain_info(self):
        return ('ResNet50_pretrained', 'https://paddlemodels.bj.bcebos.com/video_classification/ResNet50_pretrained.tar.gz')

    def weights_info(self):
        return ('stnet_kinetics', 
                'https://paddlemodels.bj.bcebos.com/video_classification/stnet_kinetics.tar.gz')

    def load_pretrain_params(self, exe, pretrain, prog, place):
        def is_parameter(var):
            if isinstance(var, fluid.framework.Parameter):
                return isinstance(var, fluid.framework.Parameter) and (not ("fc_0" in var.name)) \
                    and (not ("batch_norm" in var.name)) and (not ("xception" in var.name)) and (not ("conv3d" in var.name))

        logger.info("Load pretrain weights from {}, exclude fc, batch_norm, xception, conv3d layers.".format(pretrain))
        vars = filter(is_parameter, prog.list_vars())
        fluid.io.load_vars(exe, pretrain, vars=vars, main_program=prog)

        param_tensor = fluid.global_scope().find_var(
            "conv1_weights").get_tensor()
        param_numpy = np.array(param_tensor)
        param_numpy = np.mean(param_numpy, axis=1, keepdims=True) / self.seglen
        param_numpy = np.repeat(param_numpy, 3 * self.seglen, axis=1)
        param_tensor.set(param_numpy.astype(np.float32), place)
