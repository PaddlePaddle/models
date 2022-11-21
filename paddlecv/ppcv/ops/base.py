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
import importlib
import math
import numpy as np
try:
    from collections.abc import Sequence
except Exception:
    from collections import Sequence

import paddle
from paddle.inference import Config
from paddle.inference import create_predictor

from ppcv.ops.predictor import PaddlePredictor
from ppcv.utils.download import get_model_path

__all__ = ["BaseOp", ]


def create_operators(params, mod):
    """
    create operators based on the config

    Args:
        params(list): a dict list, used to create some operators
        mod(module) : a module that can import single ops
    """
    assert isinstance(params, list), ('operator config should be a list')
    if mod is None:
        mod = importlib.import_module(__name__)
    ops = []
    for operator in params:
        if isinstance(operator, str):
            op_name = operator
            param = {}
        else:
            assert isinstance(operator,
                              dict) and len(operator) == 1, "yaml format error"
            op_name = list(operator)[0]
            param = {} if operator[op_name] is None else operator[op_name]

        op = getattr(mod, op_name)(**param)
        ops.append(op)

    return ops


class BaseOp(object):
    """
    Base Operator, implement of prediction process
    Args
    """

    def __init__(self, model_cfg, env_cfg):
        self.model_cfg = model_cfg
        self.env_cfg = env_cfg
        self.input_keys = model_cfg["Inputs"]

    @classmethod
    def type(self):
        raise NotImplementedError

    @classmethod
    def get_output_keys(cls):
        raise NotImplementedError

    def get_input_keys(self):
        return self.input_keys

    def filter_input(self, last_outputs, input_keys):
        f_inputs = [{k: last[k] for k in input_keys} for last in last_outputs]
        return f_inputs

    def check_output(self, output, name):
        if not isinstance(output, Sequence):
            raise ValueError('The output of op: {} must be Sequence').format(
                name)
        output = output[0]
        if not isinstance(output, dict):
            raise ValueError(
                'The element of output in op: {} must be dict').format(name)
        out_keys = list(output.keys())
        for out, define in zip(out_keys, self.output_keys):
            if out != define:
                raise ValueError(
                    'The output key in op: {} is inconsistent, expect {}, but received {}'.
                    format(name, define, out))

    def set_frame(self, frame_id):
        self.frame_id = frame_id

    def __call__(self, image_list):
        raise NotImplementedError
