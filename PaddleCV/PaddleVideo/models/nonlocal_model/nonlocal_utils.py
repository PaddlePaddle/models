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
import logging
logger = logging.getLogger(__name__)


def is_parameter(var):
    return isinstance(var, fluid.framework.Parameter)


def load_pretrain_params_from_file(exe, prog, pretrained_file, place):
    """
    The pretrined_file stores ResNet50/101 parameters pretrained on ImageNet.
    However, the conv_weights of Nonlocal model is not the same as that in ResNet50/101 because the 
    input shape is [N, C, T, H, W] and the convolution kernels' shape is [Cout, Cin, Kt, Kh, Kw]. It is
    different from the convolution kernels of ResNet whose shape is typically [Cout, Cin, Kh, Kw].
    So it is recommendated to treat conv_weights specifically.
    The process is as following:
      1, check the params that will be loaded, those with the same name in the target program and pretrained_file. 
         These params will be called common params in this function.
      2, load params from the pretrained_file into a state dict, named file_state_dict in this method.
      3, get the value of common params in the file_state_dict, treat the convolution parameters specifically.
      4, set the value to params in the target program
    """

    logger.info('load pretrained params from {}'.format(pretrained_file))
    if os.path.isdir(pretrained_file):
        # get params' list in prog
        param_list = filter(is_parameter, prog.list_vars())
        param_name_list = []
        param_shape_dict = {}
        for p in param_list:
            param_name_list.append(p.name)
            param_shape_dict[p.name] = p.shape

        # get all params' names in pretrained_file
        param_name_from_file = os.listdir(pretrained_file)
        # get common params of prog and pretrained_file
        # only those common params will be loaded from pretrained_file into prog
        common_names = get_common_names(param_name_list, param_name_from_file)

        file_state_dict = fluid.load_program_state(pretrained_file)
        pretrain_state_dict = {}
        for name in common_names:
            common_array = file_state_dict[name]
            param_shape = param_shape_dict[name]
            if len(param_shape) == 5:
                # transform the loaded conv weights into the format of [Cout, Cin, Kt, Kh, Kw]
                num_inflate = param_shape[2]
                pretrain_state_dict[name] = np.stack(
                    [common_array] * num_inflate, axis=2) / float(num_inflate)
                logger.info("load inflated({}) param {}".format(num_inflate,
                                                                name))
            else:
                pretrain_state_dict[name] = common_array
                logger.info("load param {}".format(name))

        fluid.set_program_state(prog, pretrain_state_dict)
    else:
        raise TypeError(
            "pretrained file {} is not in a directory, not suitable to load params".
            format(pretrained_file))


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


def load_weights_params_from_file(exe, prog, weights, place):
    """
    The params of the training process is stored in the file named weights.
    However, the network of the training and test process is slightly different due to the layer 
    named "pred" was fc in trainng but convolution in test. When loading weights of pred (pred_w), 
    from the pretrained file, shape mismatch error will be raised due to the check in fluid.io. 
    This check on params' shape is newly added in fluid.version==1.6.0. So it is recommendated to 
    treat pred_w specifically.
    The process is as following:
      1, load the parmas from weights file into a state_dict
      2, specifically treat the paramter named "pred_w" from the foramt of fc into convolution
      3, set the state_dict to prog
    """

    logger.info('Load test weights from {}'.format(weights))

    # get the param_list in prog
    prog_vars = list(filter(is_parameter, prog.list_vars()))

    if weights[-9:] == '.pdparams':
        weights = weights[:-9]

    state_dict = fluid.load_program_state(weights, var_list=prog_vars)
    pred_array = state_dict["pred_w"]
    pred_array = np.transpose(pred_array, (1, 0))
    pred_array = np.reshape(
        pred_array, [pred_array.shape[0], pred_array.shape[1], 1, 1, 1])
    state_dict["pred_w"] = pred_array
    fluid.set_program_state(prog, state_dict)
