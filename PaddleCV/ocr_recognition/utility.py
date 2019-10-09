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


def get_ctc_feeder_data(data, place, need_label=True):
    pixel_tensor = fluid.LoDTensor()
    pixel_data = None
    pixel_data = np.concatenate(
        list(map(lambda x: x[0][np.newaxis, :], data)),
        axis=0).astype("float32")
    pixel_tensor.set(pixel_data, place)
    label_tensor = to_lodtensor(list(map(lambda x: x[1], data)), place)
    if need_label:
        return {"pixel": pixel_tensor, "label": label_tensor}
    else:
        return {"pixel": pixel_tensor}


def get_ctc_feeder_for_infer(data, place):
    return get_ctc_feeder_data(data, place, need_label=False)


def get_attention_feeder_data(data, place, need_label=True):
    pixel_tensor = fluid.LoDTensor()
    pixel_data = None
    pixel_data = np.concatenate(
        list(map(lambda x: x[0][np.newaxis, :], data)),
        axis=0).astype("float32")
    pixel_tensor.set(pixel_data, place)
    label_in_tensor = to_lodtensor(list(map(lambda x: x[1], data)), place)
    label_out_tensor = to_lodtensor(list(map(lambda x: x[2], data)), place)
    if need_label:
        return {
            "pixel": pixel_tensor,
            "label_in": label_in_tensor,
            "label_out": label_out_tensor
        }
    else:
        return {"pixel": pixel_tensor}


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


def check_gpu(use_gpu):
    """
    Log error and exit when set use_gpu=true in paddlepaddle
    cpu version.
    """
    err = "Config use_gpu cannot be set as true while you are " \
          "using paddlepaddle cpu version ! \nPlease try: \n" \
          "\t1. Install paddlepaddle-gpu to run model on GPU \n" \
          "\t2. Set use_gpu as false in config file to run " \
          "model on CPU"

    try:
        if use_gpu and not fluid.is_compiled_with_cuda():
            print(err)
            sys.exit(1)
    except Exception as e:
        pass
