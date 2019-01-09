from __future__ import print_function
import argparse
import logging
import os
import time

import numpy as np

import paddle.fluid as fluid
import reader


def parse_args():
    parser = argparse.ArgumentParser(
        description="PaddlePaddle Word2vec example")
    parser.add_argument(
        '--train_data_path',
        type=str,
        default='./data/1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled',
        help="The path of taining dataset")
    parser.add_argument(
        '--dict_path',
        type=str,
        default='./data/1-billion_dict',
        help="The path of data dict")
    parser.add_argument(
        '--with_hs',
        action='store_true',
        required=False,
        default=False,
        help='using hierarchical sigmoid, (default: False)')
    return parser.parse_args()


def GetFileList(data_path):
    return os.listdir(data_path)


def converter(args):
    filelist = GetFileList(args.train_data_path)

    word2vec_reader = reader.Word2VecReader(
        args.dict_path, args.train_data_path, filelist, 0, 1)
    word2vec_reader.async_train(args.with_hs)


if __name__ == "__main__":
    args = parse_args()
    converter(args)
