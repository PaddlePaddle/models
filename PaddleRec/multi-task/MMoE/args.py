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


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature_size", type=int, default=499, help="feature_size")
    parser.add_argument("--expert_size", type=int, default=16, help="expert_size")
    parser.add_argument("--tower_size", type=int, default=8, help="tower_size")
    parser.add_argument("--expert_num", type=int, default=8, help="expert_num")
    parser.add_argument("--gate_num", type=int, default=2, help="gate_num")
    parser.add_argument("--epochs", type=int, default=400, help="epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="batch_size")
    parser.add_argument('--use_gpu', type=int, default=0, help='whether using gpu')
<<<<<<< HEAD
    parser.add_argument('--model_dir',type=str, default='model_dir', help="model_dir")
    parser.add_argument('--train_data_path',type=str, default='train_data', help="train_data_path")
    parser.add_argument('--test_data_path',type=str, default='test_data', help="test_data_path")
=======
    parser.add_argument('--model_dir',type=str, default='./model_dir', help="model_dir")
    parser.add_argument('--train_data_path',type=str, default='./train_data', help="train_data_path")
    parser.add_argument('--test_data_path',type=str, default='./test_data', help="test_data_path")
>>>>>>> 6c87d487199034dde7d0d5dde4ef324fe5d273f1
    args = parser.parse_args()
    return args
    
def data_preparation_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_path", type=str, default='', help="train_path")
    parser.add_argument("--test_path", type=str, default='', help="test_path")

    parser.add_argument('--train_data_path',type=str, default='', help="train_data_path")
    parser.add_argument('--test_data_path',type=str, default='', help="test_data_path")
    args = parser.parse_args()
    return args
