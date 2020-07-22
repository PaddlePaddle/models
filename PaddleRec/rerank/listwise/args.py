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
    parser.add_argument("--epochs", type=int, default=20, help="epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="batch_size")
    parser.add_argument("--test_epoch", type=int, default=19, help="test_epoch")
    parser.add_argument('--use_gpu', type=int, default=0, help='whether using gpu')
    parser.add_argument('--model_dir', type=str, default='./model_dir', help='model_dir')
    parser.add_argument('--embd_dim', type=int, default=16, help='embd_dim')
    parser.add_argument('--hidden_size', type=int, default=128, help='hidden_size')
    parser.add_argument('--item_vocab', type=int, default=200, help='item_vocab')
    parser.add_argument('--user_vocab', type=int, default=200, help='user_vocab')
    parser.add_argument('--item_len', type=int, default=5, help='item_len')
    parser.add_argument('--sample_size', type=int, default=100, help='sample_size')
    parser.add_argument('--base_lr', type=float, default=0.01, help='base_lr')

    args = parser.parse_args()
    return args