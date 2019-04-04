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
import cPickle
import paddle.fluid as fluid

from ..model import ModelBase
import resnet_video

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

    def build_input(self, use_pyreader=True):
        input_shape = [3, self.video_length, self.crop_size, self.crop_size]
        label_shape = [1]
        py_reader = None
        if use_pyreader:
            assert self.mode != 'infer', \
                        'pyreader is not recommendated when infer, please set use_pyreader to be false.'
            py_reader = fluid.layers.py_reader(
                capacity=20,
                shapes=[[-1] + input_shape, [-1] + label_shape],
                dtypes=['float32', 'int64'],
                name='train_py_reader'
                if self.is_training else 'test_py_reader',
                use_double_buffer=True)
            data, label = fluid.layers.read_file(py_reader)
            self.py_reader = py_reader
        else:
            data = fluid.layers.data(
                name='train_data' if self.is_training else 'test_data',
                shape=input_shape,
                dtype='float32')
            if self.mode != 'infer':
                label = fluid.layers.data(
                    name='train_label' if self.is_training else 'test_label',
                    shape=label_shape,
                    dtype='int64')
            else:
                label = None
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

    def pretrain_info(self):
        return None, None

    def weights_info(self):
        pass

    def load_pretrain_params(self, exe, pretrain, prog, place):
        load_params_from_file(exe, prog, pretrain, place)

    def load_test_weights(self, exe, weights, prog, place):
        super(NonLocal, self).load_test_weights(exe, weights, prog, place)
        pred_w = fluid.global_scope().find_var('pred_w').get_tensor()
        pred_array = np.array(pred_w)
        pred_w_shape = pred_array.shape
        if len(pred_w_shape) == 2:
            logger.info('reshape for pred_w when test')
            pred_array = np.transpose(pred_array, (1, 0))
            pred_w_shape = pred_array.shape
            pred_array = np.reshape(
                pred_array, [pred_w_shape[0], pred_w_shape[1], 1, 1, 1])
            pred_w.set(pred_array.astype('float32'), place)


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


def load_params_from_pkl_file(prog, pretrained_file, place):
    param_list = prog.block(0).all_parameters()
    param_name_list = [p.name for p in param_list]

    if os.path.exists(pretrained_file):
        params_from_file = cPickle.load(open(pretrained_file))
        if len(params_from_file.keys()) == 1:
            params_from_file = params_from_file['blobs']
        param_name_from_file = params_from_file.keys()
        param_list = prog.block(0).all_parameters()
        param_name_list = [p.name for p in param_list]

        common_names = get_common_names(param_name_list, param_name_from_file)

        logger.info('-------- loading params -----------')
        for name in common_names:
            t = fluid.global_scope().find_var(name).get_tensor()
            t_array = np.array(t)
            f_array = params_from_file[name]
            if 'pred' in name:
                assert np.prod(t_array.shape) == np.prod(
                    f_array.shape), "number of params should be the same"
                if t_array.shape == f_array.shape:
                    logger.info("pred param is the same {}".format(name))
                else:
                    re_f_array = np.reshape(f_array, t_array.shape)
                    t.set(re_f_array.astype('float32'), place)
                    logger.info("load pred param {}".format(name))
                    continue
            if t_array.shape == f_array.shape:
                t.set(f_array.astype('float32'), place)
                logger.info("load param {}".format(name))
            elif (t_array.shape[:2] == f_array.shape[:2]) and (
                    t_array.shape[-2:] == f_array.shape[-2:]):
                num_inflate = t_array.shape[2]
                stack_f_array = np.stack(
                    [f_array] * num_inflate, axis=2) / float(num_inflate)
                assert t_array.shape == stack_f_array.shape, "inflated shape should be the same with tensor {}".format(
                    name)
                t.set(stack_f_array.astype('float32'), place)
                logger.info("load inflated({}) param {}".format(num_inflate,
                                                                name))
            else:
                logger.info("Invalid case for name: {}".format(name))
                raise
        logger.info("finished loading params from resnet pretrained model")


def load_params_from_paddle_file(exe, prog, pretrained_file, place):
    if os.path.isdir(pretrained_file):
        param_list = prog.block(0).all_parameters()
        param_name_list = [p.name for p in param_list]
        param_shape = {}
        for name in param_name_list:
            param_tensor = fluid.global_scope().find_var(name).get_tensor()
            param_shape[name] = np.array(param_tensor).shape

        param_name_from_file = os.listdir(pretrained_file)
        common_names = get_common_names(param_name_list, param_name_from_file)

        logger.info('-------- loading params -----------')

        # load params from file 
        def is_parameter(var):
            if isinstance(var, fluid.framework.Parameter):
                return isinstance(var, fluid.framework.Parameter) and \
                          os.path.exists(os.path.join(pretrained_file, var.name))

        logger.info("Load pretrain weights from file {}".format(
            pretrained_file))
        vars = filter(is_parameter, prog.list_vars())
        fluid.io.load_vars(exe, pretrained_file, vars=vars, main_program=prog)

        # reset params if necessary
        for name in common_names:
            t = fluid.global_scope().find_var(name).get_tensor()
            t_array = np.array(t)
            origin_shape = param_shape[name]
            if 'pred' in name:
                assert np.prod(t_array.shape) == np.prod(
                    origin_shape), "number of params should be the same"
                if t_array.shape == origin_shape:
                    logger.info("pred param is the same {}".format(name))
                else:
                    reshaped_t_array = np.reshape(t_array, origin_shape)
                    t.set(reshaped_t_array.astype('float32'), place)
                    logger.info("load pred param {}".format(name))
                    continue
            if t_array.shape == origin_shape:
                logger.info("load param {}".format(name))
            elif (t_array.shape[:2] == origin_shape[:2]) and (
                    t_array.shape[-2:] == origin_shape[-2:]):
                num_inflate = origin_shape[2]
                stack_t_array = np.stack(
                    [t_array] * num_inflate, axis=2) / float(num_inflate)
                assert origin_shape == stack_t_array.shape, "inflated shape should be the same with tensor {}".format(
                    name)
                t.set(stack_t_array.astype('float32'), place)
                logger.info("load inflated({}) param {}".format(num_inflate,
                                                                name))
            else:
                logger.info("Invalid case for name: {}".format(name))
                raise
        logger.info("finished loading params from resnet pretrained model")
    else:
        logger.info(
            "pretrained file is not in a directory, not suitable to load params".
            format(pretrained_file))
        pass


def get_common_names(param_name_list, param_name_from_file):
    # name check and return common names both in param_name_list and file
    common_names = []
    paddle_only_names = []
    file_only_names = []
    logger.info('-------- comon params -----------')
    for name in param_name_list:
        if name in param_name_from_file:
            common_names.append(name)
            logger.info(name)
        else:
            paddle_only_names.append(name)
    logger.info('-------- paddle only params ----------')
    for name in paddle_only_names:
        logger.info(name)
    logger.info('-------- file only params -----------')
    for name in param_name_from_file:
        if name in param_name_list:
            assert name in common_names
        else:
            file_only_names.append(name)
            logger.info(name)
    return common_names


def load_params_from_file(exe, prog, pretrained_file, place):
    logger.info('load params from {}'.format(pretrained_file))
    if '.pkl' in pretrained_file:
        load_params_from_pkl_file(prog, pretrained_file, place)
    else:
        load_params_from_paddle_file(exe, prog, pretrained_file, place)
