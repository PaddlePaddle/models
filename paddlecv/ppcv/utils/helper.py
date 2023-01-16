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

import ppcv
from ppcv.ops import *
from ppcv.core.workspace import get_global_op


def get_output_keys(cfg=None):
    op_list = get_global_op()
    if cfg is None:
        output = dict()
        for name, op in op_list.items():
            if op.type() != 'OUTPUT':
                keys = op.get_output_keys()
                output.update({name: keys})
    else:
        output = {'input.image', 'input.video'}
        for op in cfg:
            op_arch = op_list[list(op.keys())[0]]
            op_cfg = list(op.values())[0]
            if op_arch.type() == 'OUTPUT': continue
            for out_name in op_arch.get_output_keys():
                name = op_cfg['name'] + '.' + out_name
                output.add(name)
    return output


def gen_input_name(input_keys, last_ops, output_keys):
    # generate input name according to input_keys and last_ops
    # the name format is {last_ops}.{input_key}
    input_name = list()
    for key in input_keys:
        found = False
        if key in output_keys:
            found = True
            input_name.append(key)
        else:
            for op in last_ops:
                name = op + '.' + key
                if name in input_name:
                    raise ValueError("Repeat input: {}".format(name))
                if name in output_keys:
                    input_name.append(name)
                    found = True
                    break
        if not found:
            raise ValueError(
                "Input: {} could not be found from the last ops: {}. The outputs of these last ops are {}".
                format(key, last_ops, output_keys))
    return input_name
