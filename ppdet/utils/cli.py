# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import sys
from argparse import ArgumentParser, RawDescriptionHelpFormatter, REMAINDER

import yaml


class ColorTTY(object):
    def __init__(self):
        super(ColorTTY, self).__init__()
        self.colors = ['red', 'green', 'yellow', 'blue', 'magenta', 'cyan']

    def __getattr__(self, attr):
        if attr in self.colors:
            color = self.colors.index(attr) + 31

            def color_message(message):
                return "[{}m{}[0m".format(color, message)

            setattr(self, attr, color_message)
            return color_message

    def bold(self, message):
        return self.with_code('01', message)

    def with_code(self, code, message):
        return "[{}m{}[0m".format(code, message)


def parse_args():
    parser = ArgumentParser(formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("-c", "--config",            help="configuration file to use")
    parser.add_argument("-r", "--resume_checkpoint", default=None, type=str, help="The checkpoint path for resuming training.")
    # TODO(dangqingqing) remove this flag
    parser.add_argument("-f", "--fusebn",            default=True, help="Whether to fuse params of batch norm to scale and bias.")
    parser.add_argument("-o", "--opt",               nargs=REMAINDER, help="set configuration options")
    args = parser.parse_args()

    if args.config is None:
        raise ValueError("Please specify --config=configure_file_path.")

    cli_config = {}
    if 'opt' in vars(args) and args.opt is not None:
        for s in args.opt:
            s = s.strip()
            k, v = s.split('=')
            if '.' not in k:
                cli_config[k] = v
            else:
                keys = k.split('.')
                cli_config[keys[0]] = {}
                cur = cli_config[keys[0]]
                for idx, key in enumerate(keys[1:]):
                    if idx == len(keys) - 2:
                        cur[key] = yaml.load(v, Loader=yaml.Loader)
                    else:
                        cur[key] = {}
                        cur = cur[key]
    args.cli_config = cli_config
    return args
