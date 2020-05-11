#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import math
import paddle.fluid as fluid

class ArcMarginLoss():
    def __init__(self, class_dim, margin=0.15, scale=80.0, easy_margin=False):
        self.class_dim = class_dim
        self.margin = margin
        self.scale = scale
        self.easy_margin = easy_margin

    def loss(self, input, label):
        out = self.arc_margin_product(input, label, self.class_dim, self.margin, self.scale, self.easy_margin)
        #loss = fluid.layers.softmax_with_cross_entropy(logits=out, label=label)
        out = fluid.layers.softmax(input=out)
        loss = fluid.layers.cross_entropy(input=out, label=label)
        return loss, out

    def arc_margin_product(self, input, label, out_dim, m, s, easy_margin=False):
        #input = fluid.layers.l2_normalize(input, axis=1)
        input_norm = fluid.layers.sqrt(fluid.layers.reduce_sum(fluid.layers.square(input), dim=1))
        input = fluid.layers.elementwise_div(input, input_norm, axis=0)

        weight = fluid.layers.create_parameter(
                    shape=[out_dim, input.shape[1]],
                    dtype='float32',
                    name='weight_norm',
                    attr=fluid.param_attr.ParamAttr(
                                  initializer=fluid.initializer.Xavier()))
        #weight = fluid.layers.l2_normalize(weight, axis=1)
        weight_norm = fluid.layers.sqrt(fluid.layers.reduce_sum(fluid.layers.square(weight), dim=1))
        weight = fluid.layers.elementwise_div(weight, weight_norm, axis=0)
        weight = fluid.layers.transpose(weight, perm = [1, 0])
        cosine = fluid.layers.mul(input, weight)
        sine = fluid.layers.sqrt(1.0 - fluid.layers.square(cosine) + 1e-6)

        cos_m = math.cos(m)
        sin_m = math.sin(m)
        phi = cosine * cos_m - sine * sin_m

        th = math.cos(math.pi - m)
        mm = math.sin(math.pi - m) * m
        if easy_margin:
            phi = self.paddle_where_more_than(cosine, 0, phi, cosine)
        else:
            phi = self.paddle_where_more_than(cosine, th, phi, cosine-mm)

        one_hot = fluid.one_hot(input=label, depth=out_dim)
        one_hot = fluid.layers.squeeze(input=one_hot, axes=[1])
        output = fluid.layers.elementwise_mul(one_hot, phi) + fluid.layers.elementwise_mul((1.0 - one_hot), cosine)
        output = output * s
        return output

    def paddle_where_more_than(self, target, limit, x, y):
        mask = fluid.layers.cast(x=(target>limit), dtype='float32')
        output = fluid.layers.elementwise_mul(mask, x) + fluid.layers.elementwise_mul((1.0 - mask), y)
        return output
