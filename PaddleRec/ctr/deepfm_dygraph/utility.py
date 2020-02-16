from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import distutils.util

import numpy as np
import six


def parse_args():
    parser = argparse.ArgumentParser(description="deepfm dygraph")
    parser.add_argument(
        '--train_data_dir',
        type=str,
        default='data/train_data',
        help='The path of train data (default: data/train_data)')
    parser.add_argument(
        '--test_data_dir',
        type=str,
        default='data/test_data',
        help='The path of test data (default: models)')
    parser.add_argument(
        '--model_output_dir',
        type=str,
        default='models',
        help='The path for model to store (default: models)')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='',
        help='The path for model and optimizer to load (default: "")')
    parser.add_argument(
        '--feat_dict',
        type=str,
        default='data/aid_data/feat_dict_10.pkl2',
        help='The path of feat_dict')
    parser.add_argument(
        '--num_epoch',
        type=int,
        default=10,
        help="The number of epochs to train (default: 10)")
    parser.add_argument(
        '--batch_size',
        type=int,
        default=4096,
        help="The size of mini-batch (default:4096)")
    parser.add_argument(
        '--use_gpu', type=distutils.util.strtobool, default=True)

    parser.add_argument(
        '--embedding_size',
        type=int,
        default=10,
        help="The size for embedding layer (default:10)")
    parser.add_argument(
        '--layer_sizes',
        nargs='+',
        type=int,
        default=[400, 400, 400],
        help='The size of each layers (default: [400, 400, 400])')
    parser.add_argument(
        '--act',
        type=str,
        default='relu',
        help='The activation of each layers (default: relu)')
    parser.add_argument(
        '--lr', type=float, default=1e-3, help='Learning rate (default: 1e-3)')
    parser.add_argument(
        '--reg', type=float, default=1e-4, help=' (default: 1e-4)')
    parser.add_argument('--num_field', type=int, default=39)
    parser.add_argument('--num_feat', type=int, default=1086460)  # 2090493

    return parser.parse_args()


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


def to_numpy(data):
    flattened_data = np.concatenate(data, axis=0).astype("int64")
    flattened_data = flattened_data.reshape([len(flattened_data), 1])
    return flattened_data
