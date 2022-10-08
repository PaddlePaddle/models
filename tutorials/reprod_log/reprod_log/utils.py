# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import logging
import paddle
import numpy as np
from typing import Union


def init_logger(log_file=None, name='root', log_level=logging.DEBUG):
    logger = logging.getLogger(name)

    formatter = logging.Formatter(
        '[%(asctime)s] %(name)s %(levelname)s: %(message)s',
        datefmt="%Y/%m/%d %H:%M:%S")

    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    if log_file is not None:
        dir_name = os.path.dirname(log_file)
        if len(dir_name) > 0 and not os.path.exists(dir_name):
            os.makedirs(dir_name)
        file_handler = logging.FileHandler(log_file, 'w')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    logger.setLevel(log_level)
    return logger


def np2torch(data: dict):
    import torch

    assert isinstance(data, dict)
    torch_input = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            torch_input[k] = torch.Tensor(v)
        else:
            torch_input[k] = v
    return torch_input


def np2paddle(data: dict):
    assert isinstance(data, dict)
    paddle_input = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            paddle_input[k] = paddle.Tensor(v)
        else:
            paddle_input[k] = v
    return paddle_input


def paddle2np(data: Union[paddle.Tensor, dict]=None):
    if isinstance(data, dict):
        np_data = {}
        for k, v in data.items():
            np_data[k] = v.numpy()
        return np_data
    else:
        return {'output': data.numpy()}


def torch2np(data):
    if isinstance(data, dict):
        np_data = {}
        for k, v in data.items():
            np_data[k] = v.detach().numpy()
        return np_data
    else:
        return {'output': data.detach().numpy()}


def check_print_diff(diff_dict,
                     diff_method='mean',
                     diff_threshold: float=1e-6,
                     print_func=print,
                     indent: str='\t',
                     level: int=0):
    """
    对 diff 字典打印并进行检查的函数

    :param diff_dict:
    :param diff_method: 检查diff的函数，目前支持 min,max,mean,all四种形式，并且支持min,max,mean的相互组合成的list形式，如['min','max']
    :param diff_threshold:
    :param print_func:
    :param indent:
    :param level:
    :return:
    """
    if level == 0:
        if isinstance(diff_method, str):
            if diff_method == 'all':
                diff_method = ['min', 'max', 'mean']
            else:
                diff_method = [diff_method]
    for method in diff_method:
        assert method in ['all', 'min', 'max', 'mean']

    all_passed = True
    cur_indent = indent * level
    for k, v in diff_dict.items():
        if 'mean' in v and 'min' in v and 'max' in v and len(v) == 3:
            print_func('{}{}: '.format(cur_indent, k))
            sub_passed = True
            for method in diff_method:
                if v[method] > diff_threshold:
                    sub_passed = False
                print_func("{}{} diff: check passed: {}, value: {}".format(
                    cur_indent + indent, method, sub_passed, v[method]))
                all_passed = all_passed and sub_passed
        else:
            print_func('{}{}'.format(cur_indent, k))
            sub_passed = check_print_diff(v, diff_method, diff_threshold,
                                          print_func, indent, level + 1)
            all_passed = all_passed and sub_passed
    return all_passed
