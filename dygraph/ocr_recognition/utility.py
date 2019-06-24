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
import distutils.util
import numpy as np
import paddle.fluid as fluid
import six


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


def to_numpy(data):

    flattened_data = np.concatenate(data, axis=0).astype("int64")
    flattened_data = flattened_data.reshape([len(flattened_data), 1])
    return flattened_data


def get_attention_feeder_data(data, need_label=True):
    pixel_data = None
    pixel_data = np.concatenate(
        list(map(lambda x: x[0][np.newaxis, :], data)),
        axis=0).astype("float32")

    label_in = to_numpy(list(map(lambda x: x[1], data)))
    label_out = to_numpy(list(map(lambda x: x[2], data)))

    mask = list(map(lambda x: x[3], data))
    mask = np.concatenate(mask, axis=0)

    if need_label:
        return {
            "pixel": pixel_data,
            "label_in": label_in,
            "label_out": label_out,
            'mask': mask,
        }
    else:
        return {"pixel": pixel_data}


def get_attention_feeder_for_infer(data, place):
    batch_size = len(data)
    init_ids_data = np.array([0 for _ in range(batch_size)], dtype='int64')
    init_scores_data = np.array(
        [1. for _ in range(batch_size)], dtype='float32')
    init_ids_data = init_ids_data.reshape((batch_size, 1))
    init_scores_data = init_scores_data.reshape((batch_size, 1))
    init_recursive_seq_lens = [1] * batch_size
    init_recursive_seq_lens = [init_recursive_seq_lens, init_recursive_seq_lens]
    init_ids = fluid.create_lod_tensor(init_ids_data, init_recursive_seq_lens,
                                       place)
    init_scores = fluid.create_lod_tensor(init_scores_data,
                                          init_recursive_seq_lens, place)

    pixel_tensor = fluid.LoDTensor()
    pixel_data = None
    pixel_data = np.concatenate(
        list(map(lambda x: x[0][np.newaxis, :], data)),
        axis=0).astype("float32")
    pixel_tensor.set(pixel_data, place)
    return {
        "pixel": pixel_tensor,
        "init_ids": init_ids,
        "init_scores": init_scores
    }
