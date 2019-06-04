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
from .roidb_source import RoiDbSource
from .simple_source import SimpleSource


def build(config):
    """ build dataset from source data, 
        default source type is 'RoiDbSource'
    """
    data_cf = {k.lower(): v for k, v in config['data_cf'].items()}
    data_cf['cname2cid'] = config['cname2cid']
    args = copy.deepcopy(data_cf)
    if data_cf['type'] in ['VOCSource', 'COCOSource', 'RoiDbSource']:
        source_type = 'RoiDbSource'
    else:
        source_type = data_cf['type']
    del args['type']
    if source_type == 'RoiDbSource':
        return RoiDbSource(**args)
    elif source_type == 'SimpleSource':
        return SimpleSource(**args)
    else:
        raise ValueError('not supported source type[%s]' % (source_type))
