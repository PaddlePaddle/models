# -*- coding: utf-8 -*-
#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
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

import argparse
import distutils.util


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset_prefix", type=str, help="file prefix for train data")

    parser.add_argument(
        "--optimizer",
        type=str,
        default='adam',
        help="optimizer to use, only supprt[sgd|adam]")

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="learning rate for optimizer")

    parser.add_argument(
        "--num_layers",
        type=int,
        default=1,
        help="layers number of encoder and decoder")
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=256,
        help="hidden size of encoder and decoder")
    parser.add_argument("--vocab_size", type=int, help="source vocab size")

    parser.add_argument(
        "--batch_size", type=int, help="batch size of each step")

    parser.add_argument(
        "--max_epoch", type=int, default=20, help="max epoch for the training")

    parser.add_argument(
        "--max_len",
        type=int,
        default=1280,
        help="max length for source and target sentence")
    parser.add_argument(
        "--dec_dropout_in",
        type=float,
        default=0.5,
        help="decoder input drop probability")
    parser.add_argument(
        "--dec_dropout_out",
        type=float,
        default=0.5,
        help="decoder output drop probability")
    parser.add_argument(
        "--enc_dropout_in",
        type=float,
        default=0.,
        help="encoder input drop probability")
    parser.add_argument(
        "--enc_dropout_out",
        type=float,
        default=0.,
        help="encoder output drop probability")
    parser.add_argument(
        "--word_keep_prob",
        type=float,
        default=0.5,
        help="word keep probability")
    parser.add_argument(
        "--init_scale",
        type=float,
        default=0.0,
        help="init scale for parameter")
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=5.0,
        help="max grad norm for global norm clip")

    parser.add_argument(
        "--model_path",
        type=str,
        default='model',
        help="model path for model to save")

    parser.add_argument(
        "--reload_model", type=str, help="reload model to inference")

    parser.add_argument(
        "--infer_output_file",
        type=str,
        default='infer_output.txt',
        help="file name for inference output")

    parser.add_argument(
        "--beam_size", type=int, default=10, help="file name for inference")

    parser.add_argument(
        '--use_gpu',
        type=bool,
        default=False,
        help='Whether using gpu [True|False]')

    parser.add_argument(
        "--enable_ce",
        action='store_true',
        help="The flag indicating whether to run the task "
        "for continuous evaluation.")

    parser.add_argument(
        "--profile", action='store_true', help="Whether enable the profile.")

    parser.add_argument(
        "--warm_up",
        type=int,
        default=10,
        help='number of warm up epochs for KL')

    parser.add_argument(
        "--kl_start", type=float, default=0.1, help='KL start value, upto 1.0')

    parser.add_argument(
        "--attr_init",
        type=str,
        default='normal_initializer',
        help="initializer for paramters")

    parser.add_argument(
        "--cache_num", type=int, default=1, help='cache num for reader')

    parser.add_argument(
        "--max_decay",
        type=int,
        default=5,
        help='max decay tries (if exceeds, early stop)')

    parser.add_argument(
        "--sort_cache",
        action='store_true',
        help='sort cache before batch to accelerate training')

    args = parser.parse_args()
    return args
