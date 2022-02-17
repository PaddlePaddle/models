#  Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

import paddle.fluid as fluid
from paddle.fluid import ParamAttr
import numpy as np


class ShiftingAttentionModel(object):
    """Shifting Attention Model"""

    def __init__(self, input_dim, seg_num, n_att, name):
        self.n_att = n_att
        self.input_dim = input_dim
        self.seg_num = seg_num
        self.name = name
        self.gnorm = np.sqrt(n_att)

    def softmax_m1(self, x):
        x_shape = fluid.layers.shape(x)
        x_shape.stop_gradient = True
        flat_x = fluid.layers.reshape(x, shape=(-1, self.seg_num))
        flat_softmax = fluid.layers.softmax(flat_x)
        return fluid.layers.reshape(flat_softmax, shape=x_shape)

    def glorot(self, n):
        return np.sqrt(1.0 / np.sqrt(n))

    def forward(self, x):
        """Forward shifting attention model.

        Args:
          x: input features in shape of [N, L, F].

        Returns:
          out: output features in shape of [N, F * C]
        """

        trans_x = fluid.layers.transpose(x, perm=[0, 2, 1])
        # scores and weight in shape [N, C, L], sum(weights, -1) = 1
        trans_x = fluid.layers.unsqueeze(trans_x, [-1])
        scores = fluid.layers.conv2d(
            trans_x,
            self.n_att,
            filter_size=1,
            param_attr=ParamAttr(
                name=self.name + ".conv.weight",
                initializer=fluid.initializer.MSRA(uniform=False)),
            bias_attr=ParamAttr(
                name=self.name + ".conv.bias",
                initializer=fluid.initializer.MSRA()))
        scores = fluid.layers.squeeze(scores, [-1])
        weights = self.softmax_m1(scores)

        glrt = self.glorot(self.n_att)
        self.w = fluid.layers.create_parameter(
            shape=(self.n_att, ),
            dtype=x.dtype,
            attr=ParamAttr(self.name + ".shift_w"),
            default_initializer=fluid.initializer.Normal(0.0, glrt))
        self.b = fluid.layers.create_parameter(
            shape=(self.n_att, ),
            dtype=x.dtype,
            attr=ParamAttr(name=self.name + ".shift_b"),
            default_initializer=fluid.initializer.Normal(0.0, glrt))

        outs = []
        for i in range(self.n_att):
            # slice weight and expand to shape [N, L, C]
            weight = fluid.layers.slice(
                weights, axes=[1], starts=[i], ends=[i + 1])
            weight = fluid.layers.transpose(weight, perm=[0, 2, 1])
            weight = fluid.layers.expand(weight, [1, 1, self.input_dim])

            w_i = fluid.layers.slice(self.w, axes=[0], starts=[i], ends=[i + 1])
            b_i = fluid.layers.slice(self.b, axes=[0], starts=[i], ends=[i + 1])
            shift = fluid.layers.reduce_sum(x * weight, dim=1) * w_i + b_i

            l2_norm = fluid.layers.l2_normalize(shift, axis=-1)
            outs.append(l2_norm / self.gnorm)

        out = fluid.layers.concat(outs, axis=1)
        return out
