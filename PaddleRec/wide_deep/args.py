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
import sys


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=40, help="epochs")
    parser.add_argument("--batch_size", type=int, default=40, help="batch_size")
    parser.add_argument('--use_gpu', type=int, default=0, help='whether using gpu')
    parser.add_argument('--test_epoch', type=str, default='1',help='test_epoch')
    parser.add_argument('--train_path', type=str, default='data/adult.data', help='train_path')
    parser.add_argument('--test_path', type=str, default='data/adult.test', help='test_path')
    parser.add_argument('--train_data_path', type=str, default='train_data/train_data.csv', help='train_data_path')
    parser.add_argument('--test_data_path', type=str, default='test_data/test_data.csv', help='test_data_path')
    parser.add_argument('--model_dir', type=str, default='model_dir', help='test_data_path')
    parser.add_argument('--hidden1_units', type=int, default=75, help='hidden1_units')
    parser.add_argument('--hidden2_units', type=int, default=50, help='hidden2_units')
    parser.add_argument('--hidden3_units', type=int, default=25, help='hidden3_units')

    args = parser.parse_args()
    return args
    


