from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import six
import argparse
import functools
import distutils.util


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


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    add_arg = functools.partial(add_arguments, argparser=parser)

    # yapf: disable
    add_arg('model_net', str, "cgan", "The model used.")
    add_arg('dataset', str, "mnist", "The dataset used.")
    add_arg('data_dir', str, "./data", "The dataset root directory")
    add_arg('data_list', str, "train.txt", "The dataset list file name")
    add_arg('batch_size', int, 1, "Minibatch size.")
    add_arg('epoch', int, 200, "The number of epoch to be trained.")
    add_arg('n_gen_res', int, 9,
            "The number of resnet block when generator's network is resnet")
    add_arg('g_base_dims', int, 64, "Base channels in CycleGAN generator")
    add_arg('d_base_dims', int, 64, "Base channels in CycleGAN discriminator")
    add_arg('d_nlayers', int, 3,
            "only used when CycleGAN discriminator is nlayers")
    add_arg('load_size', int, 286, "the image size when load the image")
    add_arg('crop_type', str, 'Centor',
            "the crop type, choose = ['Centor', 'Random']")
    add_arg('crop_size', int, 256, "crop size when preprocess image")
    add_arg('save_checkpoints', bool, True, "Whether to save checkpoints.")
    add_arg('run_test', bool, True, "Whether to run test.")
    add_arg('run_ce', bool, False, "Whether to run ce")
    add_arg('use_gpu', bool, True, "Whether to use GPU to train.")
    add_arg('profile', bool, False, "Whether to profile.")
    add_arg('dropout', bool, False, "Whether to use drouput.")
    add_arg('use_dropout', bool, False, "Whether to use dropout")
    add_arg('drop_last', bool, False,
            "Whether to drop the last images that cannot form a batch")
    add_arg('shuffle', bool, True, "Whether to shuffle data")
    add_arg('output', str, "./output",
            "The directory the model and the test result to be saved to.")
    add_arg('init_model', str, None, "The init model file of directory.")
    add_arg('norm_type', str, "batch_norm", "Which normalization to used")
    add_arg(
        'net_G', str, "resnet_9block",
        "Choose the CycleGAN generator's network, choose in [resnet_9block|resnet_6block|unet_128|unet_256]"
    )
    add_arg(
        'net_D', str, "nlayers",
        "Choose the CycleGAN discriminator's network, choose in [basic|nlayers|pixel]"
    )
    add_arg('learning_rate', int, 0.0002, "the initialize learning rate")
    add_arg('num_generator_time', int, 1,
            "the generator run times in training each epoch")
    add_arg('num_discriminator_time', int, 1,
            "the discrimitor run times in training each epoch")
    add_arg('noise_size', int, 100, "the noise dimension")
    add_arg('print_freq', int, 10, "the frequency of print loss")
    # yapf: enable

    args = parser.parse_args()

    return args
