# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved. 
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

import os
import sys
import numpy as np
import math
import glob
import paddle

parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
sys.path.insert(0, parent_path)

from argparse import ArgumentParser
import ppcv
from ppcv.ops import *
from ppcv.utils.helper import get_output_keys
import yaml


def argsparser():
    parser = ArgumentParser()
    parser.add_argument(
        "--config", type=str, default=None, help=("Path of configure"))
    return parser


def check_cfg_output(cfg, output_dict):
    with open(cfg) as f:
        cfg = yaml.safe_load(f)
    model_cfg = cfg['MODEL']
    output_set = {'image', 'video', 'fn'}
    for v in output_dict.values():
        for name in v:
            output_set.add(name)
    for ops in model_cfg:
        op_name = list(ops.keys())[0]
        cfg_dict = list(ops.values())[0]
        cfg_input = cfg_dict['Inputs']
        for key in cfg_input:
            key = key.split('.')[-1]
            assert key in output_set, "Illegal input: {} in {}.".format(
                key, op_name)


def check_name(cfg):
    config = None
    config = vars(cfg)['config']
    output_dict = get_output_keys()
    buffer = yaml.dump(output_dict)
    print('----------- Op output names ---------')
    print(buffer)
    if config is not None:
        check_cfg_output(config, output_dict)


if __name__ == '__main__':
    parser = argsparser()
    FLAGS = parser.parse_args()
    check_name(FLAGS)
