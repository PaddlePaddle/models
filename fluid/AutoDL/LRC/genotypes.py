# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved
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
#
# Based on:
# --------------------------------------------------------
# DARTS
# Copyright (c) 2018, Hanxiao Liu.
# Licensed under the Apache License, Version 2.0;
# --------------------------------------------------------

from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

PRIMITIVES = [
    'none', 'max_pool_3x3', 'avg_pool_3x3', 'skip_connect', 'sep_conv_3x3',
    'sep_conv_5x5', 'dil_conv_3x3', 'dil_conv_5x5'
]

NASNet = Genotype(
    normal=[
        ('sep_conv_5x5', 1),
        ('sep_conv_3x3', 0),
        ('sep_conv_5x5', 0),
        ('sep_conv_3x3', 0),
        ('avg_pool_3x3', 1),
        ('skip_connect', 0),
        ('avg_pool_3x3', 0),
        ('avg_pool_3x3', 0),
        ('sep_conv_3x3', 1),
        ('skip_connect', 1),
    ],
    normal_concat=[2, 3, 4, 5, 6],
    reduce=[
        ('sep_conv_5x5', 1),
        ('sep_conv_7x7', 0),
        ('max_pool_3x3', 1),
        ('sep_conv_7x7', 0),
        ('avg_pool_3x3', 1),
        ('sep_conv_5x5', 0),
        ('skip_connect', 3),
        ('avg_pool_3x3', 2),
        ('sep_conv_3x3', 2),
        ('max_pool_3x3', 1),
    ],
    reduce_concat=[4, 5, 6], )

AmoebaNet = Genotype(
    normal=[
        ('avg_pool_3x3', 0),
        ('max_pool_3x3', 1),
        ('sep_conv_3x3', 0),
        ('sep_conv_5x5', 2),
        ('sep_conv_3x3', 0),
        ('avg_pool_3x3', 3),
        ('sep_conv_3x3', 1),
        ('skip_connect', 1),
        ('skip_connect', 0),
        ('avg_pool_3x3', 1),
    ],
    normal_concat=[4, 5, 6],
    reduce=[
        ('avg_pool_3x3', 0),
        ('sep_conv_3x3', 1),
        ('max_pool_3x3', 0),
        ('sep_conv_7x7', 2),
        ('sep_conv_7x7', 0),
        ('avg_pool_3x3', 1),
        ('max_pool_3x3', 0),
        ('max_pool_3x3', 1),
        ('conv_7x1_1x7', 0),
        ('sep_conv_3x3', 5),
    ],
    reduce_concat=[3, 4, 6])

DARTS_V1 = Genotype(
    normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 0),
            ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1),
            ('sep_conv_3x3', 0), ('skip_connect', 2)],
    normal_concat=[2, 3, 4, 5],
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2),
            ('max_pool_3x3', 0), ('max_pool_3x3', 0), ('skip_connect', 2),
            ('skip_connect', 2), ('avg_pool_3x3', 0)],
    reduce_concat=[2, 3, 4, 5])
DARTS_V2 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0),
            ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('skip_connect', 0),
            ('skip_connect', 0), ('dil_conv_3x3', 2)],
    normal_concat=[2, 3, 4, 5],
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2),
            ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2),
            ('skip_connect', 2), ('max_pool_3x3', 1)],
    reduce_concat=[2, 3, 4, 5])

MY_DARTS = Genotype(
    normal=[('sep_conv_3x3', 0), ('skip_connect', 1), ('skip_connect', 0),
            ('dil_conv_5x5', 1), ('skip_connect', 0), ('sep_conv_3x3', 1),
            ('skip_connect', 0), ('sep_conv_3x3', 1)],
    normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0),
            ('skip_connect', 2), ('max_pool_3x3', 0), ('skip_connect', 2),
            ('skip_connect', 2), ('skip_connect', 3)],
    reduce_concat=range(2, 6))

DARTS = MY_DARTS
