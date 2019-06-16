#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import os
import shutil
import logging
logger = logging.getLogger(__name__)

import paddle.fluid as fluid

from .download import get_weights_path

__all__ = ['load', 'save']


def is_url(path):
    """
    Whether path is URL.
    Args:
        path (string): URL string or not.
    """
    return path.startswith('http://') or path.startswith('https://')


def load(exe, prog, path):
    """
    Load model from the given path.
    Args:
        exe (fluid.Executor): The fluid.Executor object.
        prog (fluid.Program): load weight to which Program object.
        path (string): URL string or loca model path.
    """
    if is_url(path):
        path = get_weights_path(path)

    logger.info('Loading model from {}...'.format(path))

    def if_exist(var):
        b = os.path.exists(os.path.join(path, var.name))
        if b:
            logger.debug('load weight {}'.format(var.name))
        return b

    fluid.io.load_vars(exe, path, prog, predicate=if_exist)


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
    fluid.io.save_persistables(exe, path, prog)
