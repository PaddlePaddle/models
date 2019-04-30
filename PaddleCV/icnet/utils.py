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
import six
import paddle.fluid as fluid

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


def to_lodtensor(data, place):
    seq_lens = [len(seq) for seq in data]
    cur_len = 0
    lod = [cur_len]
    for l in seq_lens:
        cur_len += l
        lod.append(cur_len)
    flattened_data = np.concatenate(data, axis=0).astype("int32")
    flattened_data = flattened_data.reshape([len(flattened_data), 1])
    res = fluid.LoDTensor()
    res.set(flattened_data, place)
    res.set_lod([lod])
    return res


def get_feeder_data(data, place, for_test=False):
    feed_dict = {}
    image_t = fluid.LoDTensor()
    image_t.set(data[0], place)
    feed_dict["image"] = image_t

    if not for_test:
        labels_sub1_t = fluid.LoDTensor()
        labels_sub2_t = fluid.LoDTensor()
        labels_sub4_t = fluid.LoDTensor()
        mask_sub1_t = fluid.LoDTensor()
        mask_sub2_t = fluid.LoDTensor()
        mask_sub4_t = fluid.LoDTensor()

        labels_sub1_t.set(data[1], place)
        labels_sub2_t.set(data[3], place)
        mask_sub1_t.set(data[2], place)
        mask_sub2_t.set(data[4], place)
        labels_sub4_t.set(data[5], place)
        mask_sub4_t.set(data[6], place)
        feed_dict["label_sub1"] = labels_sub1_t
        feed_dict["label_sub2"] = labels_sub2_t
        feed_dict["mask_sub1"] = mask_sub1_t
        feed_dict["mask_sub2"] = mask_sub2_t
        feed_dict["label_sub4"] = labels_sub4_t
        feed_dict["mask_sub4"] = mask_sub4_t
    else:
        label_t = fluid.LoDTensor()
        mask_t = fluid.LoDTensor()
        label_t.set(data[1], place)
        mask_t.set(data[2], place)
        feed_dict["label"] = label_t
        feed_dict["mask"] = mask_t

    return feed_dict
