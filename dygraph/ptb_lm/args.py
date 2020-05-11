#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
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

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model_type",
        type=str,
        default="small",
        help="model_type [test|small|medium|large]")
    parser.add_argument(
        "--rnn_model",
        type=str,
        default="static",
        help="model_type [static|padding|cudnn]")
    parser.add_argument(
        "--data_path", type=str, help="all the data for train,valid,test")
    parser.add_argument('--para_init', action='store_true')
    parser.add_argument(
        '--use_gpu', type=str2bool, default=True, help='whether using gpu')
    parser.add_argument(
        '--log_path',
        help='path of the log file. If not set, logs are printed to console')
    parser.add_argument(
        '--save_model_dir',
        type=str,
        default="models",
        help='dir of the saved model.')
    parser.add_argument(
        '--init_from_pretrain_model',
        type=str,
        default=None,
        help='dir to init model.')
    # NOTE: used for benchmark
    parser.add_argument(
        '--max_iter',
        type=int,
        default=0,
        help='the max iters for train, used for benchmark.')
    parser.add_argument('--ce', action='store_true', help="run ce")
    args = parser.parse_args()
    return args
