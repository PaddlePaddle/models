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
"""
Parse argument.
"""

import argparse
import json


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


class HParams(dict):
    """ Hyper-parameters class

    Store hyper-parameters in training / infer / ... scripts.
    """

    def __getattr__(self, name):
        if name in self.keys():
            return self[name]
        for v in self.values():
            if isinstance(v, HParams):
                if name in v:
                    return v[name]
        raise AttributeError(f"'HParams' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        self[name] = value

    def save(self, filename):
        with open(filename, "w", encoding="utf-8") as fp:
            json.dump(self, fp, ensure_ascii=False,
                      indent=4, sort_keys=False)

    def load(self, filename):
        with open(filename, "r", encoding="utf-8") as fp:
            params_dict = json.load(fp)
        for k, v in params_dict.items():
            if isinstance(v, dict):
                self[k].update(HParams(v))
            else:
                self[k] = v


def parse_args(parser):
    """ Parse hyper-parameters from cmdline. """
    parsed = parser.parse_args()
    args = HParams()
    optional_args = parser._action_groups[1]
    for action in optional_args._group_actions[1:]:
        arg_name = action.dest
        args[arg_name] = getattr(parsed, arg_name)
    for group in parser._action_groups[2:]:
        group_args = HParams()
        for action in group._group_actions:
            arg_name = action.dest
            group_args[arg_name] = getattr(parsed, arg_name)
        if len(group_args) > 0:
            args[group.title] = group_args
    return args
