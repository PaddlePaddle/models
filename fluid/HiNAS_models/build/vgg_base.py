# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved
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

import paddle.fluid as fluid
from absl import flags

import build.layers as layers
import build.ops as _ops

FLAGS = flags.FLAGS
flags.DEFINE_integer("num_stages", 5, "number of stages")
flags.DEFINE_integer("width", 64, "network width")

num_classes = 10

ops = [
    _ops.conv_1x1,  #0
    _ops.conv_2x2,  #1
    _ops.conv_3x3,  #2
    _ops.dilated_2x2,  #3
    _ops.conv_1x2_2x1,  #4
    _ops.conv_1x3_3x1,  #5
    _ops.sep_2x2,  #6
    _ops.sep_3x3,  #7
    _ops.maxpool_2x2,  #8
    _ops.maxpool_3x3,
    _ops.avgpool_2x2,  #10
    _ops.avgpool_3x3,
]


def net(inputs, tokens):
    depth = len(tokens)
    q, r = divmod(depth + 1, FLAGS.num_stages)
    downsample_steps = [
        i * q + max(0, i + r - FLAGS.num_stages + 1) - 2
        for i in range(1, FLAGS.num_stages)
    ]

    x = layers.conv(inputs, FLAGS.width, (3, 3))
    x = layers.bn_relu(x)

    for i, token in enumerate(tokens):
        downsample = i in downsample_steps
        x = ops[token](x, downsample)
        print("%s \t-> shape %s" % (ops[token].__name__, x.shape))
        if downsample:
            print("=" * 12)
        x = layers.bn_relu(x)

    x = layers.global_avgpool(x)
    x = layers.dropout(x)
    logits = layers.fully_connected(x, num_classes)

    return fluid.layers.softmax(logits)
