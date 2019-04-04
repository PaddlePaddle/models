"""Contains common utility functions."""
#  Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import six
import time
import subprocess
import distutils.util
import numpy as np

from paddle.fluid import core


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
    print("-----------  Configuration Arguments -----------")
    for arg, value in sorted(six.iteritems(vars(args))):
        print("%s: %s" % (arg, value))
    print("------------------------------------------------")


def add_arguments(argname, type, default, help, argparser, **kwargs):
    """Add argparse's argument.

    Usage:

    .. code-block:: python

        parser = argparse.ArgumentParser()
        add_argument("name", str, "Jonh", "User name.", parser)
        args = parser.parse_args()
    """
    type = distutils.util.strtobool if type == bool else type
    argparser.add_argument(
        "--" + argname,
        default=default,
        type=type,
        help=help + ' Default: %(default)s.',
        **kwargs)

def fmt_time():
    """ get formatted time for now
    """
    now_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    return now_str

def recall_topk(fea, lab, k = 1):
    fea = np.array(fea)
    fea = fea.reshape(fea.shape[0], -1)
    n = np.sqrt(np.sum(fea**2, 1)).reshape(-1, 1)
    fea = fea / n
    a = np.sum(fea ** 2, 1).reshape(-1, 1)
    b = a.T
    ab = np.dot(fea, fea.T)
    d = a + b - 2*ab
    d = d + np.eye(len(fea)) * 1e8
    sorted_index = np.argsort(d, 1)
    res = 0
    for i in range(len(fea)):
        pred = lab[sorted_index[i][0]]
        if lab[i] == pred:
            res += 1.0
    res = res / len(fea)
    return res

def get_gpu_num():
    visibledevice = os.getenv('CUDA_VISIBLE_DEVICES')
    if visibledevice:
        devicenum = len(visibledevice.split(','))
    else:
        devicenum = subprocess.check_output(
            [str.encode('nvidia-smi'), str.encode('-L')]).decode('utf-8').count('\n')
    return devicenum

