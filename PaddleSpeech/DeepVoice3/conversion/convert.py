#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
import torch

import paddle
from paddle import fluid

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--pytorch-model",
    dest='pytorch_model',
    type=str,
    help="The source pytorch mode.")
parser.add_argument(
    "--paddle-model",
    dest='paddle_model',
    type=str,
    help="The directory to save paddle model, now saves model as a folder.")
parser.add_argument(
    "--name-map",
    dest="name_map",
    type=str,
    help="name mapping for the source model and the target model.")


def read_name_map(fname):
    """
    There should be a 3-column file.
    The first comuln is the name of parameter in pytorch model's state dict;
    The second column is the name of parameter in paddle model's state dict;
    The third column is the shape of the repective parameter in paddle model.
    """
    name_map = {}
    with open(fname, 'rt') as f:
        for line in f:
            src_key, tgt_key, tgt_shape = line.strip().split('\t')
            tgt_shape = eval(tgt_shape)
            name_map[src_key] = (tgt_key, tgt_shape)
    return name_map


def torch2paddle(state_dict, name_map, dirname):
    """
    state_dict: pytorch model's state dict.
    name_map: a text file for name mapping from pytorch model to paddle model.
    dirname: path of the paddle model to save.
    """
    program = fluid.Program()
    global_block = program.global_block()

    for k in state_dict.keys():
        global_block.create_parameter(
            name=name_map[k][0],
            shape=[1],
            dtype='float32',
            initializer=fluid.initializer.Constant(value=0.0))

    place = fluid.core.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())
    exe.run(program)

    # NOTE: transpose the pytorch model's parameter if neccessary
    # we do not transpose here because we used conv instead of FC layer to replace Linear in pytorch,
    # which does not need us to transpose the paramerters.
    # but when you use a FC layer corresponding a torch Linear module, be sure to transpose the weight.
    # Other transformations are not concerned, but users should check the data shape to ensure that
    # the transformations are what's expected.
    for k, v in state_dict.items():
        fluid.global_scope().find_var(name_map[k][0]).get_tensor().set(
            v.cpu().numpy().reshape(name_map[k][1]), place)
    fluid.io.save_params(exe, dirname, main_program=program)


if __name__ == "__main__":
    args, _ = parser.parse_known_args()
    result = torch.load(args.pytorch_model)
    state_dict = result["state_dict"]
    name_map = read_name_map(args.name_map)
    torch2paddle(state_dict, name_map, args.paddle_model)
