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

from .download import download_weights

__all__ = ['load', 'save']


def is_url(path):
    """
    Whether path is URL.
    Args:
        path (string): URL string or not.
    """
    return path.startswith('http://')


def load(exe, path):
    """
    Load model from the given path.
    Args:
        path (string): URL string or loca model path.
    """
    if is_url(path):
        path = download_weights(path)

    logger.info('Load model from {}.'.format(path))

    def if_exist(var):
        return os.path.exists(os.path.join(path, var.name))

    fluid.io.load_vars(exe, path, predicate=if_exist)


def save(exe, path):
    """
    Load model from the given path.
    Args:
        path (string): URL string or loca model path.
    """
    if os.path.isdir(path):
        shutil.rmtree(path)
    logger.info('Save model to {}.'.format(path))
    fluid.io.save_persistables(exe, path)
