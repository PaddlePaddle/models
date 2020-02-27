#copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
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
import sys
import six
import argparse
import functools
import distutils.util
import trainer


def print_arguments(args):
    ''' Print argparse's argument
    Usage:

    .. code-block:: python

        parser = argparse.ArgumentParser()
        parser.add_argument("name", default="Jonh", type=str, help="User name.")
        args = parser.parse_args()
        print_arguments(args)

    :param args: Input argparse.Namespace for printing.
    :type args: argparse.Namespace
    '''
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


def base_parse_args(parser):
    add_arg = functools.partial(add_arguments, argparser=parser)
    # yapf: disable
    add_arg('model_net', str, "CGAN", "The model used.")
    add_arg('dataset', str, "mnist", "The dataset used.")
    add_arg('data_dir', str, "./data", "The dataset root directory")
    add_arg('train_list', str, None, "The train list file name")
    add_arg('test_list', str, None, "The test list file name")
    add_arg('batch_size', int, 1, "Minibatch size.")
    add_arg('epoch', int, 200, "The number of epoch to be trained.")
    add_arg('g_base_dims', int, 64, "Base channels in generator")
    add_arg('d_base_dims', int, 64, "Base channels in discriminator")
    add_arg('image_size', int, 286, "the image size when load the image")
    add_arg('crop_type', str, 'Centor',
            "the crop type, choose = ['Centor', 'Random']")
    add_arg('crop_size', int, 256, "crop size when preprocess image")
    add_arg('save_checkpoints', bool, True, "Whether to save checkpoints.")
    add_arg('run_test', bool, True, "Whether to run test.")
    add_arg('use_gpu', bool, True, "Whether to use GPU to train.")
    add_arg('profile', bool, False, "Whether to profile.")

    # NOTE: add args for profiler, used for benchmark
    add_arg('profiler_path', str, '/tmp/profile', "the  profiler output files. (used for benchmark)")
    add_arg('max_iter', int, 0, "the max iter to train. (used for benchmark)")

    add_arg('dropout', bool, False, "Whether to use drouput.")
    add_arg('drop_last', bool, False,
            "Whether to drop the last images that cannot form a batch")
    add_arg('shuffle', bool, True, "Whether to shuffle data")
    add_arg('output', str, "./output",
            "The directory the model and the test result to be saved to.")
    add_arg('init_model', str, None, "The init model file of directory.")
    add_arg('gan_mode', str, "vanilla", "The init model file of directory.")
    add_arg('norm_type', str, "batch_norm", "Which normalization to used")
    add_arg('learning_rate', float, 0.0002, "the initialize learning rate")
    add_arg('lambda_L1', float, 100.0, "the initialize lambda parameter for L1 loss")
    add_arg('num_generator_time', int, 1,
            "the generator run times in training each epoch")
    add_arg('num_discriminator_time', int, 1,
            "the discriminator run times in training each epoch")
    add_arg('print_freq', int, 10, "the frequency of print loss")
    # yapf: enable

    return parser


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser = base_parse_args(parser)
    cfg, _ = parser.parse_known_args()
    model_name = cfg.model_net
    model_cfg = trainer.get_special_cfg(model_name)
    parser = model_cfg(parser)
    args = parser.parse_args()
    return args
