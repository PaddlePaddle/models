#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
import math
import distutils.util
import numpy as np
import inspect
import matplotlib
import six
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

img_dim = 28


def get_parent_function_name():
    return inspect.stack()[2][3] + '.' + inspect.stack()[1][3] + '.' + str(
        inspect.stack()[2][2]) + '.'


def plot(gen_data):
    pad_dim = 1
    paded = pad_dim + img_dim
    gen_data = gen_data.reshape(gen_data.shape[0], img_dim, img_dim)
    n = int(math.ceil(math.sqrt(gen_data.shape[0])))
    gen_data = (np.pad(
        gen_data, [[0, n * n - gen_data.shape[0]], [pad_dim, 0], [pad_dim, 0]],
        'constant').reshape((n, n, paded, paded)).transpose((0, 2, 1, 3))
                .reshape((n * paded, n * paded)))
    fig = plt.figure(figsize=(8, 8))
    plt.axis('off')
    plt.imshow(gen_data, cmap='Greys_r', vmin=-1, vmax=1)
    return fig


def check(a):
    a = np.sort(np.array(a).flatten())
    return [
        np.average(a), np.min(a), np.max(a), a[int(len(a) * 0.25)],
        a[int(len(a) * 0.75)]
    ]


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
