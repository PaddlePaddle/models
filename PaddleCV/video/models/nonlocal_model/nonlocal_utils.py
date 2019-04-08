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


def load_params_from_file(exe, prog, pretrained_file, place):
    logger.info('load params from {}'.format(pretrained_file))
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
