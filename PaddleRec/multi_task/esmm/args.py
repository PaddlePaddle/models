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
    parser.add_argument("--epochs", type=int, default=100, help="epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="batch_size")
    parser.add_argument("--embed_size", type=int, default=12, help="embed_size")
    parser.add_argument("--cpu_num", type=int, default=2, help="cpu_num")
    parser.add_argument('--use_gpu', type=int, default=0, help='whether using gpu')
    parser.add_argument('--model_dir', type=str, default='./model_dir', help='whether using gpu')
    
    parser.add_argument('--train_data_path', type=str, default='./train_data', help='train_data_path')
    parser.add_argument('--test_data_path', type=str, default='./test_data', help='test_data_path')
    parser.add_argument('--vocab_path', type=str, default='./vocab_size.txt', help='vocab_path')
    parser.add_argument("--train_sample_size", type=int, default=sys.maxsize, help="train_sample_size")
    parser.add_argument("--test_sample_size", type=int, default=sys.maxsize, help="test_sample_size")
    
    
    args = parser.parse_args()
    return args
    


