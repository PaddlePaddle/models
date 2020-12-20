
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import distutils.util
import sys

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    # -------------Data & Model Path-------------
    parser.add_argument(
        '--train_files_path',
        type=str,
        default='./train_data_full',
        help="The path of training dataset")
    parser.add_argument(
        '--test_files_path',
        type=str,
        default='./test_data_full',
        help="The path of testing dataset")
    parser.add_argument(
        '--model_dir',
        type=str,
        default='./model_dir',
        help='The path for model to store (default: models)')

    parser.add_argument(
        '--test_epoch',
        type=str,
        default='10',
        help='test_epoch')

    # -------------Training parameter-------------
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.001,
        help="Initial learning rate for training")
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1000,
        help="The size of mini-batch (default:1000)")
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of epochs for training.")
    parser.add_argument(
        "--reduction_ratio",
        type=int,
        default=3,
        help="reduction_ratio")
    parser.add_argument(
        '--bilinear_type',
        type=str,
        default='all',
        help="bilinear_type")
    parser.add_argument(
        "--dropout_rate",
        type=int,
        default=0.5,
        help="dropout_rate")

    # -------------Network parameter-------------
    parser.add_argument(
        '--embedding_size',
        type=int,
        default=10,
        help="The size for embedding layer (default:10)")
    parser.add_argument(
        '--sparse_feature_dim',
        type=int,
        default=1000001,
        help='sparse feature hashing space for index processing')
    parser.add_argument(
        '--dense_feature_dim',
        type=int,
        default=13,
        help='dense feature shape')

    # -------------device parameter-------------
    parser.add_argument(
        '--use_gpu',
        type=int,
        default=0,
        help='use_gpu')


    return parser.parse_args()