#!/usr/bin/env python
# -*- coding: utf-8 -*-
######################################################################
#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
######################################################################
"""
File: args.py
"""

from __future__ import print_function

import six
import argparse


# define argument parser & add common arguments
def base_parser():
    parser = argparse.ArgumentParser(description="Arguments for running classifier.")
    parser.add_argument(
        '--epoch',
        type=int,
        default=100,
        help='Number of epoches for training. (default: %(default)d)')
    parser.add_argument(
        '--task_name',
        type=str,
        default='match',
        help='task name for training')
    parser.add_argument(
        '--max_seq_len',
        type=int,
        default=512,
        help='Number of word of the longest seqence. (default: %(default)d)')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=8096,
        help='Total token number in batch for training. (default: %(default)d)')
    parser.add_argument(
        '--voc_size',
        type=int,
        default=14373,
        help='Total token number in batch for training. (default: %(default)d)')
    parser.add_argument(
        '--init_checkpoint',
        type=str,
        default=None,
        help='init checkpoint to resume training from. (default: %(default)s)')
    parser.add_argument(
        '--save_inference_model_path',
        type=str,
        default="inference_model",
        help='save inference model. (default: %(default)s)')
    parser.add_argument(
        '--output',
        type=str,
        default="./output/pred.txt",
        help='init checkpoint to resume training from. (default: %(default)s)')
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1e-2,
        help='Learning rate used to train with warmup. (default: %(default)f)')
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=0.01,
        help='Weight decay rate for L2 regularizer. (default: %(default)f)')
    parser.add_argument(
        '--checkpoints',
        type=str,
        default="checkpoints",
        help='Path to save checkpoints. (default: %(default)s)')
    parser.add_argument(
        '--vocab_path',
        type=str,
        default=None,
        help='Vocabulary path. (default: %(default)s)')
    parser.add_argument(
        '--data_dir',
        type=str,
        default="./real_data",
        help='Path of training data. (default: %(default)s)')
    parser.add_argument(
        '--skip_steps',
        type=int,
        default=10,
        help='The steps interval to print loss. (default: %(default)d)')
    parser.add_argument(
        '--save_steps',
        type=int,
        default=10000,
        help='The steps interval to save checkpoints. (default: %(default)d)')
    parser.add_argument(
        '--validation_steps',
        type=int,
        default=1000,
        help='The steps interval to evaluate model performance on validation '
        'set. (default: %(default)d)')
    parser.add_argument(
        '--use_cuda', action='store_true', help='If set, use GPU for training.')
    parser.add_argument(
        '--use_fast_executor',
        action='store_true',
        help='If set, use fast parallel executor (in experiment).')
    parser.add_argument(
        '--do_lower_case',
        type=bool,
        default=True,
        choices=[True, False],
        help="Whether to lower case the input text. Should be True for uncased "
        "models and False for cased models.")
    parser.add_argument(
        '--warmup_proportion',
        type=float,
        default=0.1,
        help='proportion warmup. (default: %(default)f)')
    args = parser.parse_args()
    return args

def print_arguments(args): 
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(six.iteritems(vars(args))):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')

