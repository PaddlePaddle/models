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
LayerNorm layer.
"""

# from paddle.fluid.dygraph import LayerNorm

from six.moves import reduce

import paddle.fluid as fluid
import paddle.fluid.layers as layers
from paddle.fluid.dygraph import Layer
import logging

class LayerNorm(Layer):
    """ Implement LayerNorm in dygraph mode. """

    def __init__(self,
                 name_scope,
                 scale=True,
                 shift=True,
                 begin_norm_axis=1,
                 epsilon=1e-05,
                 param_attr=None,
                 bias_attr=None,
                 act=None):
        super().__init__(name_scope)
        self._scale = scale
        self._shift = shift
        self._begin_norm_axis = begin_norm_axis
        self._epsilon = epsilon
        self._param_attr = param_attr
        self._bias_attr = bias_attr
        self._act = act
        return

    def _build_once(self, input):
        """ Create parameters. """
        self._dtype = self._helper.input_dtype(input)
        input_shape = input.shape
        param_shape = [
            reduce(lambda x, y: x * y, input_shape[self._begin_norm_axis:])
        ]
        if self._scale:
            self._scale_w = self.create_parameter(
                attr=self._param_attr,
                shape=param_shape,
                dtype=self._dtype,
                default_initializer=fluid.initializer.Constant(1.0))
        else:
            if self._param_attr:
                logging.warn("param_attr are only avaliable with scale is True")

        if self._shift:
            assert self._bias_attr is not False
            self._bias_w = self.create_parameter(
                attr=self._bias_attr,
                shape=param_shape,
                dtype=self._dtype,
                is_bias=True)
        else:
            if self._bias_attr:
                logging.warn("bias_attr are only avaliable with shift is True")
        return

    def forward(self, x):
        """ Forward process of LayerNorm. """
        mean = layers.reduce_mean(x,
                                  dim=list(range(self._begin_norm_axis, len(x.shape))),
                                  keep_dim=True)
        shift_x = layers.elementwise_sub(x=x, y=mean, axis=0)
        variance = layers.reduce_mean(layers.square(shift_x),
                                      dim=list(range(self._begin_norm_axis, len(x.shape))),
                                      keep_dim=True)
        r_stdev = layers.rsqrt(variance + self._epsilon)
        norm_x = layers.elementwise_mul(x=shift_x, y=r_stdev, axis=0)
        out = layers.elementwise_mul(x=norm_x, y=self._scale_w, axis=-1)
        out = layers.elementwise_add(x=out, y=self._bias_w, axis=-1)
        return out
