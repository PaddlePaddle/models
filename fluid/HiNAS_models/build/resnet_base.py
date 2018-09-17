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

flags.DEFINE_integer("num_stages", 3, "number of stages")
flags.DEFINE_integer("num_blocks", 5, "number of blocks per stage")
flags.DEFINE_integer("num_ops", 2, "number of operations per block")
flags.DEFINE_integer("width", 64, "network width")
flags.DEFINE_string("downsample", "pool", "conv or pool")

num_classes = 10

ops = [
    _ops.conv_1x1,
    _ops.conv_2x2,
    _ops.conv_3x3,
    _ops.dilated_2x2,
    _ops.conv_1x2_2x1,
    _ops.conv_1x3_3x1,
    _ops.sep_2x2,
    _ops.sep_3x3,
    _ops.maxpool_2x2,
    _ops.maxpool_3x3,
    _ops.avgpool_2x2,
    _ops.avgpool_3x3,
]


def net(inputs, tokens):
    """ build network with skip links """

    x = layers.conv(inputs, FLAGS.width, (3, 3))

    num_ops = FLAGS.num_blocks * FLAGS.num_ops
    x = stage(x, tokens[:num_ops], pre_activation=True)
    for i in range(1, FLAGS.num_stages):
        x = stage(x, tokens[i * num_ops:(i + 1) * num_ops], downsample=True)

    x = layers.bn_relu(x)
    x = layers.global_avgpool(x)
    x = layers.dropout(x)
    logits = layers.fully_connected(x, num_classes)

    return fluid.layers.softmax(logits)


def stage(x, tokens, pre_activation=False, downsample=False):
    """ build network's stage. Stage consists of blocks """

    x = block(x, tokens[:FLAGS.num_ops], pre_activation, downsample)
    for i in range(1, FLAGS.num_blocks):
        print("-" * 12)
        x = block(x, tokens[i * FLAGS.num_ops:(i + 1) * FLAGS.num_ops])
    print("=" * 12)

    return x


def block(x, tokens, pre_activation=False, downsample=False):
    """ build block. """

    if pre_activation:
        x = layers.bn_relu(x)
        res = x
    else:
        res = x
        x = layers.bn_relu(x)

    x = ops[tokens[0]](x, downsample)
    print("%s \t-> shape %s" % (ops[0].__name__, x.shape))
    for token in tokens[1:]:
        x = layers.bn_relu(x)
        x = ops[token](x)
        print("%s \t-> shape %s" % (ops[token].__name__, x.shape))

    if downsample:
        filters = res.shape[1]
        if FLAGS.downsample == "conv":
            res = layers.conv(res, filters * 2, (1, 1), (2, 2))
        elif FLAGS.downsample == "pool":
            res = layers.avgpool(res, (2, 2), (2, 2))
            res = fluid.layers.pad(res, (0, 0, filters // 2, filters // 2, 0, 0,
                                         0, 0))
        else:
            raise NotImplementedError

    return x + res
