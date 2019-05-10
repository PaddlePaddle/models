"""
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
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import copy
from .loader import load
from .roidb_source import RoiDbSource

def build(config):
    """ build dataset from source data, 
        default source type is 'RoiDbSource'
    """
    args = copy.deepcopy(config)
    if 'type' in config:
        source_type = config['type']
        del args['type']
    else:
        source_type = 'RoiDbSource'

    if source_type == 'RoiDbSource':
        return RoiDbSource(**args)
    else:
        raise ValueError('not supported source type[%s]' % (source_type))

