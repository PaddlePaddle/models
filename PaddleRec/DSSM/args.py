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
    parser.add_argument("--epochs", type=int, default=100, help="epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="batch_size")
    parser.add_argument('--use_gpu', type=bool, default=False, help='whether using gpu')
    parser.add_argument('--TRIGRAM_D', type=int, default=1000, help='TRIGRAM_D')
    parser.add_argument('--L1_N', type=int, default=300, help='L1_N')
    parser.add_argument('--L2_N', type=int, default=300, help='L2_N')
    parser.add_argument('--L3_N', type=int, default=128, help='L3_N')
    parser.add_argument('--Neg', type=int, default=4, help='Neg')
    parser.add_argument('--base_lr', type=int, default=0.01, help='base_lr')
    parser.add_argument('--model_dir', type=str, default="./model_dir", help='model_dir')
    args = parser.parse_args()
    return args

