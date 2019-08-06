#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
"""
Auto Dialogue Evaluation.
"""

import argparse
import six


def parse_args():
    """
    Auto Dialogue Evaluation Config
    """
    parser = argparse.ArgumentParser('Automatic Dialogue Evaluation.')
    parser.add_argument(
        '--do_train',
        type=bool,
        default=False,
        help='Whether to perform training.')
    parser.add_argument(
        '--do_val',
        type=bool,
        default=False,
        help='Whether to perform evaluation.')
    parser.add_argument(
        '--do_infer',
        type=bool,
        default=False,
        help='Whether to perform inference.')
    parser.add_argument(
        '--loss_type', type=str, default='CLS', help='Loss type, CLS or L2.')

    #data path
    parser.add_argument(
        '--train_path', type=str, default=None, help='Path of training data')
    parser.add_argument(
        '--val_path', type=str, default=None, help='Path of validation data')
    parser.add_argument(
        '--test_path', type=str, default=None, help='Path of test data')
    parser.add_argument(
        '--save_path', type=str, default='tmp', help='Save path')

    #step fit for data size
    parser.add_argument('--print_step', type=int, default=50, help='Print step')
    parser.add_argument('--save_step', type=int, default=400, help='Save step')
    parser.add_argument(
        '--num_scan_data', type=int, default=20, help='Save step')

    parser.add_argument(
        '--word_emb_init',
        type=str,
        default=None,
        help='Path to the initial word embedding')
    parser.add_argument(
        '--init_model', type=str, default=None, help='Path to the init model')

    parser.add_argument(
        '--use_cuda',
        action='store_true',
        help='If set, use cuda for training.')
    parser.add_argument(
        '--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument(
        '--hidden_size', type=int, default=256, help='Hidden size')
    parser.add_argument(
        '--emb_size', type=int, default=256, help='Embedding size')
    parser.add_argument(
        '--vocab_size', type=int, default=484016, help='Vocabulary size')
    parser.add_argument(
        '--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument(
        '--sample_pro',
        type=float,
        default=1,
        help='Sample probability for training data')
    parser.add_argument(
        '--max_len', type=int, default=50, help='Max length for sentences')

    args = parser.parse_args()
    return args


def print_arguments(args):
    """
    Print Config
    """
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(six.iteritems(vars(args))):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')
