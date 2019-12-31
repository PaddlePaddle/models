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
"""
Contains common utility functions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import six
import logging
import numpy as np
import paddle.fluid as fluid

__all__ = ["check_gpu",  "print_arguments", "parse_outputs", "Stat"]

logger = logging.getLogger(__name__)


def check_gpu(use_gpu):
    """
    Log error and exit when set use_gpu=True in paddlepaddle
    cpu version.
    """
    err = "Config use_gpu cannot be set as True while you are " \
          "using paddlepaddle cpu version ! \nPlease try: \n" \
          "\t1. Install paddlepaddle-gpu to run model on GPU \n" \
          "\t2. Set --use_gpu=False to run model on CPU"

    try:
        if use_gpu and not fluid.is_compiled_with_cuda():
            logger.error(err)
            sys.exit(1)
    except Exception as e:
        pass


def print_arguments(args):
    """Print argparse's arguments.

    Usage:

    .. code-block:: python

        parser = argparse.ArgumentParser()
        parser.add_argument("name", default="Jonh", type=str, help="User name.")
        args = parser.parse_args()
        print_arguments(args)

    :param args: Input argparse.Namespace for printing.
    :type args: argparse.Namespace
    """
    logger.info("-----------  Configuration Arguments -----------")
    for arg, value in sorted(six.iteritems(vars(args))):
        logger.info("%s: %s" % (arg, value))
    logger.info("------------------------------------------------")


def parse_outputs(outputs, filter_key=None, extra_keys=None, prog=None):
    keys, values = [], []
    for k, v in outputs.items():
        if filter_key is not None and k.find(filter_key) < 0:
            continue
        keys.append(k)
        v.persistable = True
        values.append(v.name)

    if prog is not None and extra_keys is not None:
        for k in extra_keys:
            try:
                v = fluid.framework._get_var(k, prog)
                keys.append(k)
                v.persistable = True
                values.append(v.name)
            except:
                pass
    return keys, values


class Stat(object):
    def __init__(self):
        self.stats = {}

    def update(self, keys, values):
        for k, v in zip(keys, values):
            if k not in self.stats:
                self.stats[k] = []
            self.stats[k].append(v)

    def reset(self):
        self.stats = {}

    def get_mean_log(self):
        log = ""
        for k, v in self.stats.items():
            log += "avg_{}: {:.4f}, ".format(k, np.mean(v))
        return log
