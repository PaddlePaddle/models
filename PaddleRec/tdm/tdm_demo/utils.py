# -*- coding=utf-8 -*-
"""
#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
"""

import os
import json
import argparse
import logging
import numpy as np
import paddle.fluid as fluid
from args import print_arguments, parse_args

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("fluid")
logger.setLevel(logging.INFO)


def read_list(path):
    list = []
    with open(path, 'r') as fin:
        for line in fin.readlines():
            data = (line.split('\n'))[0].split(',')
            data = [int(i) for i in data]
            list.append(data)
    return list


def read_layer_list(path):
    layer_list = []
    layer_list_flat = []
    with open(path, 'r') as fin:
        for line in fin.readlines():
            l = []
            layer = (line.split('\n'))[0].split(',')
            layer = [int(i) for i in layer]
            for node in layer:
                if node:
                    l.append(node)
                    layer_list_flat.append(node)
            layer_list.append(l)

    layer_array = np.array(layer_list_flat)
    layer_array = layer_array.reshape([-1, 1])
    return layer_list, layer_array


def tdm_sampler_prepare(args):
    """load tdm tree param from list file"""
    travel_list = read_list(args.tree_travel_init_path)
    travel_array = np.array(travel_list)

    layer_list, layer_array = read_layer_list(args.tree_layer_init_path)
    return [travel_list, travel_array, layer_list, layer_array]


def tdm_child_prepare(args):
    """load tdm tree param from list file"""
    info_list = read_list(args.tree_info_init_path)
    info_array = np.array(info_list)
    return info_list, info_array


def trace_var(var, msg_prefix, var_name, need_trace=False, need_detail=False):
    """trace var"""
    summarize_level = 20
    if need_detail:
        summarize_level = -1
    if need_trace:
        if isinstance(var, list):
            for i, v in enumerate(var):
                fluid.layers.Print(v,
                                   message="{}[{}.{}]".format(
                                       msg_prefix, var_name, i),
                                   summarize=summarize_level)
        else:
            fluid.layers.Print(var, message="{}[{}]".format(
                msg_prefix, var_name), summarize=summarize_level)


if __name__ == "__main__":
    args = parse_args()
    print_arguments(args)
