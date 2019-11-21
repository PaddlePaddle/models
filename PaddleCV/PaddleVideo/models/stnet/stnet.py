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
            'STNET_final.pdparams',
            'https://paddlemodels.bj.bcebos.com/video_classification/STNET_final.pdparams'
        )

    def load_pretrain_params(self, exe, pretrain, prog, place):
        """
        The pretrained params are ResNet50 pretrained on ImageNet.
        However, conv1_weights of StNet is not the same as that in ResNet50 because the input are super-image
        concatanated by a series of images. When loading conv1_weights from the pretrained file, shape
        mismatch error will be raised due to the check in fluid.io. This check on params' shape is newly
        added in fluid.version==1.6.0. So it is recommendated to treat conv1_weights specifically.
        The process is as following:
          1, load params except conv1_weights from pretrain
          2, create var named 'conv1_weights' in new_scope, and load the value from the pretrain file
          3, get the value of conv1_weights in the new_scope and transform it
          4, set the transformed value to conv1_weights in prog
        """

        def is_parameter(var):
            if isinstance(var, fluid.framework.Parameter):
                return isinstance(var, fluid.framework.Parameter) and (not ("fc_0" in var.name)) \
                    and (not ("batch_norm" in var.name)) and (not ("xception" in var.name)) \
                    and (not ("conv3d" in var.name)) and (not ("conv1_weights") in var.name)

        logger.info(
            "Load pretrain weights from {}, exclude conv1, fc, batch_norm, xception, conv3d layers.".
            format(pretrain))

        # loaded params from pretrained file exclued conv1, fc, batch_norm, xception, conv3d
        prog_vars = filter(is_parameter, prog.list_vars())
        fluid.io.load_vars(exe, pretrain, vars=prog_vars, main_program=prog)

        # get global scope and conv1_weights' details
        global_scope = fluid.global_scope()
        global_block = prog.global_block()
        conv1_weights_name = "conv1_weights"
        var_conv1_weights = global_block.var(conv1_weights_name)
        tensor_conv1_weights = global_scope.var(conv1_weights_name).get_tensor()

        var_type = var_conv1_weights.type
        var_dtype = var_conv1_weights.dtype
        var_shape = var_conv1_weights.shape
        assert var_shape[
            1] == 3 * self.seglen, "conv1_weights.shape[1] shoud be 3 x seglen({})".format(
                self.seglen)
        # transform shape to be consistent with conv1_weights of ResNet50
        var_shape = (var_shape[0], 3, var_shape[2], var_shape[3])

        # create new_scope and new_prog to create var with transformed shape
        cpu_place = fluid.CPUPlace()
        exe_cpu = fluid.Executor(cpu_place)
        new_scope = fluid.Scope()
        new_prog = fluid.Program()
        new_start_prog = fluid.Program()
        new_block = new_prog.global_block()
        with fluid.scope_guard(new_scope):
            with fluid.program_guard(new_prog, new_start_prog):
                new_var = new_block.create_var(
                    name=conv1_weights_name,
                    type=var_type,
                    shape=var_shape,
                    dtype=var_dtype,
                    persistable=True)

        # load conv1_weights from pretrain file into the var created in new_scope
        with fluid.scope_guard(new_scope):
            fluid.io.load_vars(
                exe_cpu, pretrain, main_program=new_prog, vars=[new_var])

        # get the valued of loaded conv1_weights, and transform it
        new_tensor = new_scope.var(conv1_weights_name).get_tensor()
        new_value = np.array(new_tensor)
        param_numpy = np.mean(new_value, axis=1, keepdims=True) / self.seglen
        param_numpy = np.repeat(param_numpy, 3 * self.seglen, axis=1)
        # set the value of conv1_weights in the original program
        tensor_conv1_weights.set(param_numpy.astype(np.float32), place)

        # All the expected pretrained params are set to prog now
