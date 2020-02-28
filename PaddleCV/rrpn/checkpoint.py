# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import errno
import os
import shutil
import time
import numpy as np
import re
import paddle.fluid as fluid
import logging
logger = logging.getLogger(__name__)


def _load_state(path):
    if os.path.exists(path + '.pdopt'):
        # XXX another hack to ignore the optimizer state
        tmp = tempfile.mkdtemp()
        dst = os.path.join(tmp, os.path.basename(os.path.normpath(path)))
        shutil.copy(path + '.pdparams', dst + '.pdparams')
        state = fluid.io.load_program_state(dst)
        shutil.rmtree(tmp)
    else:
        state = fluid.io.load_program_state(path)
    return state


def load_params(exe, prog, path):
    """
    Load model from the given path.
    Args:
        exe (fluid.Executor): The fluid.Executor object.
        prog (fluid.Program): load weight to which Program object.
        path (string): URL string or loca model path.
    """

    if not os.path.exists(path):
        raise ValueError("Model pretrain path {} does not "
                         "exists.".format(path))

    logger.info('Loading parameters from {}...'.format(path))

    def _if_exist(var):
        param_exist = os.path.exists(os.path.join(path, var.name))
        do_load = param_exist
        if do_load:
            logger.debug('load weight {}'.format(var.name))
        return do_load

    fluid.io.load_vars(exe, path, prog, predicate=_if_exist)


def save(exe, prog, path):
    """
    Load model from the given path.
    Args:
        exe (fluid.Executor): The fluid.Executor object.
        prog (fluid.Program): save weight from which Program object.
        path (string): the path to save model.
    """
    if os.path.isdir(path):
        shutil.rmtree(path)
    logger.info('Save model to {}.'.format(path))
    fluid.save(prog, path)


def load_and_fusebn(exe, prog, path):
    """
    Fuse params of batch norm to scale and bias.
    Args:
        exe (fluid.Executor): The fluid.Executor object.
        prog (fluid.Program): save weight from which Program object.
        path (string): the path to save model.
    """
    logger.info('Load model and fuse batch norm if have from {}...'.format(
        path))

    if not os.path.exists(path):
        raise ValueError("Model path {} does not exists.".format(path))

    # Since the program uses affine-channel, there is no running mean and var
    # in the program, here append running mean and var.
    # NOTE, the params of batch norm should be like:
    #  x_scale
    #  x_offset
    #  x_mean
    #  x_variance
    #  x is any prefix
    mean_variances = set()
    bn_vars = []

    state = None
    if os.path.exists(path + '.pdparams'):
        state = _load_state(path)

    def check_mean_and_bias(prefix):
        m = prefix + 'mean'
        v = prefix + 'variance'
        if state:
            return v in state and m in state
        else:
            return (os.path.exists(os.path.join(path, m)) and
                    os.path.exists(os.path.join(path, v)))

    has_mean_bias = True

    with fluid.program_guard(prog, fluid.Program()):
        for block in prog.blocks:
            ops = list(block.ops)
            if not has_mean_bias:
                break
            for op in ops:
                if op.type == 'affine_channel':
                    # remove 'scale' as prefix
                    scale_name = op.input('Scale')[0]  # _scale
                    bias_name = op.input('Bias')[0]  # _offset
                    prefix = scale_name[:-5]
                    mean_name = prefix + 'mean'
                    variance_name = prefix + 'variance'
                    if not check_mean_and_bias(prefix):
                        has_mean_bias = False
                        break

                    bias = block.var(bias_name)

                    mean_vb = block.create_var(
                        name=mean_name,
                        type=bias.type,
                        shape=bias.shape,
                        dtype=bias.dtype)
                    variance_vb = block.create_var(
                        name=variance_name,
                        type=bias.type,
                        shape=bias.shape,
                        dtype=bias.dtype)

                    mean_variances.add(mean_vb)
                    mean_variances.add(variance_vb)

                    bn_vars.append(
                        [scale_name, bias_name, mean_name, variance_name])

    if state:
        fluid.io.set_program_state(prog, state)
    else:
        load_params(exe, prog, path)
    if not has_mean_bias:
        logger.warning(
            "There is no paramters of batch norm in model {}. "
            "Skip to fuse batch norm. And load paramters done.".format(path))
        return

    eps = 1e-5
    for names in bn_vars:
        scale_name, bias_name, mean_name, var_name = names

        scale = fluid.global_scope().find_var(scale_name).get_tensor()
        bias = fluid.global_scope().find_var(bias_name).get_tensor()
        mean = fluid.global_scope().find_var(mean_name).get_tensor()
        var = fluid.global_scope().find_var(var_name).get_tensor()

        scale_arr = np.array(scale)
        bias_arr = np.array(bias)
        mean_arr = np.array(mean)
        var_arr = np.array(var)

        bn_std = np.sqrt(np.add(var_arr, eps))
        new_scale = np.float32(np.divide(scale_arr, bn_std))
        new_bias = bias_arr - mean_arr * new_scale

        # fuse to scale and bias in affine_channel
        scale.set(new_scale, exe.place)
        bias.set(new_bias, exe.place)
