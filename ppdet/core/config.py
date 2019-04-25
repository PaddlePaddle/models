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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import yaml
from .edict import AttrDict


__all__ = ['load_cfg', 'merge_cfg']


def load_cfg(cfg_file, fmt='yaml'):
    """ Load config from given config filename, configs will be 
    loaded as AttrDict. 

    Currently supported format: ['yaml']
    """
    assert os.path.exists(cfg_file), \
        "Config file {} not exist.".format(cfg_file)
    assert fmt in ['yaml'], \
        "Format {} not supported.".format(fmt)

    with open(cfg_file) as f:
        try:
            cfg = yaml.load(f)
        except:
            return None

    return AttrDict(cfg)

def _merge_cfg_a_to_b(a, b):
    if a is None or b is None:
        return

    for k, v in a.items():
        try:
            v = eval(v)
        except:
            pass

        if b.has_key(k):
            if type(v) == type(b[k]):
                if isinstance(v, dict):
                    _merge_cfg_a_to_b(v, b[k])
                else:
                    b[k] = v
            elif type(v) in [list, tuple] and \
                    type(b[k]) in [list, tuple]:
                        b[k] = tuple(v)
            else:
                TypeError("Type mismatch for key {}".format(k))
        else:
            b[k] = v

def merge_cfg(cfg_from, cfg_to):
    """Merge config cfg_from to cfg_to, cfg_to has a higher priority.
    """

    assert isinstance(cfg_from, dict), \
            "cfg_from should be a dict"
    assert isinstance(cfg_to, dict), \
            "cfg_to should be a dict"

    _merge_cfg_a_to_b(cfg_from, cfg_to)

