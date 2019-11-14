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
Helpful functions.
"""

import numpy as np
import paddle.fluid as fluid
import paddle.fluid.layers as layers


def unsqueeze(input, axes):
    """ Implement unsqueeze in dygraph mode. """
    # return layers.unsqueeze(input, axes)
    # op:unsqueeze has bug in dygraph
    axes = [axis if axis >= 0 else axis + len(input.shape) + 1 for axis in axes]
    axes = sorted(axes, reverse=True)
    shape = list(input.shape)
    for axis in axes:
        shape.insert(axis, 1)
    return layers.reshape(input, shape)


def gumbel_softmax(input, tau=1, eps=1e-10):
    """ Basic implement of gumbel_softmax. """
    U = fluid.dygraph.to_variable(np.random.rand(*input.shape))
    # U = layers.uniform_random(input.shape, dtype=input.dtype, min=0.0, max=1.0)
    # U.stop_gradient = True
    gumbel = 0.0 - layers.log(eps - layers.log(U + eps))
    y = input + gumbel
    return layers.softmax(y / tau)


def equal(x, y, dtype=None):
    """ Implement equal in dygraph mode. """
    # if not isinstance(y, fluid.framework.Variable):
    #     y = layers.fill_constant(x.shape, x.dtype, y)
    # return layers.cast(layers.equal(x, y), dtype)
    if dtype is None:
        dtype = "float32"
    if isinstance(x, fluid.framework.Variable):
        x = x.numpy()
    if isinstance(y, fluid.framework.Variable):
        y = y.numpy()
    out = np.equal(x, y).astype(dtype)
    return fluid.dygraph.to_variable(out)


def not_equal(x, y, dtype=None):
    """ Implement not_equal in dygraph mode. """
    return 1 - equal(x, y, dtype)


def dropout(x, p):
    """ Implement dropout function like tensorflow/pytorch. """
    return layers.dropout(x, p, dropout_implementation="upscale_in_train")
