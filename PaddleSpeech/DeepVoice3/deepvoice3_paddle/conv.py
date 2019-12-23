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

import math
import numpy as np

import paddle
from paddle import fluid
import paddle.fluid.dygraph as dg

from deepvoice3_paddle.weight_norm import Conv2D, Conv2DTranspose


class Conv1D(dg.Layer):
    """
    A convolution 1D block implemented with Conv2D. Form simplicity and 
    ensuring the output has the same length as the input, it does not allow 
    stride > 1.
    """

    def __init__(self,
                 name_scope,
                 in_channels,
                 num_filters,
                 filter_size=3,
                 dilation=1,
                 groups=None,
                 causal=False,
                 param_attr=None,
                 bias_attr=None,
                 use_cudnn=True,
                 act=None,
                 dtype="float32"):
        super(Conv1D, self).__init__(name_scope, dtype=dtype)

        if causal:
            padding = dilation * (filter_size - 1)
        else:
            padding = (dilation * (filter_size - 1)) // 2

        self.in_channels = in_channels
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.dilation = dilation
        self.causal = causal
        self.padding = padding
        self.act = act

        self.conv = Conv2D(
            self.full_name(),
            num_filters=num_filters,
            filter_size=(1, filter_size),
            stride=(1, 1),
            dilation=(1, dilation),
            padding=(0, padding),
            groups=groups,
            param_attr=param_attr,
            bias_attr=bias_attr,
            use_cudnn=use_cudnn,
            act=act,
            dtype=dtype)

    def forward(self, x):
        """
        Args:
            x (Variable): Shape(B, C_in, 1, T), the input, where C_in means
                input channels.

        Returns:
            x (Variable): Shape(B, C_out, 1, T), the outputs, where C_out means
                output channels (num_filters).
        """
        x = self.conv(x)
        if self.filter_size > 1:
            if self.causal:
                x = fluid.layers.slice(
                    x, axes=[3], starts=[0], ends=[-self.padding])
            elif self.filter_size % 2 == 0:
                x = fluid.layers.slice(x, axes=[3], starts=[0], ends=[-1])
        return x

    def start_new_sequence(self):
        self.temp_weight = None
        self.input_buffer = None

    def add_input(self, x):
        """
        Adding input for a time step and compute an output for a time step.
        
        Args:
            x (Variable): Shape(B, C_in, 1, T), the input, where C_in means
                input channels, and T = 1.

        Returns:
            out (Variable): Shape(B, C_out, 1, T), the outputs, where C_out
            means output channels (num_filters), and T = 1.
            
        """
        if self.temp_weight is None:
            self.temp_weight = self._reshaped_weight()

        window_size = 1 + (self.filter_size - 1) * self.dilation
        batch_size = x.shape[0]
        in_channels = x.shape[1]

        if self.filter_size > 1:
            if self.input_buffer is None:
                self.input_buffer = fluid.layers.fill_constant(
                    [batch_size, in_channels, 1, window_size - 1],
                    dtype=x.dtype,
                    value=0.0)
            else:
                self.input_buffer = self.input_buffer[:, :, :, 1:]
            self.input_buffer = fluid.layers.concat(
                [self.input_buffer, x], axis=3)
            x = self.input_buffer
            if self.dilation > 1:
                if not hasattr(self, "indices"):
                    self.indices = dg.to_variable(
                        np.arange(0, window_size, self.dilation))
                tmp = fluid.layers.transpose(
                    self.input_buffer, perm=[3, 1, 2, 0])
                tmp = fluid.layers.gather(tmp, index=self.indices)
                tmp = fluid.layers.transpose(tmp, perm=[3, 1, 2, 0])
                x = tmp
        inputs = fluid.layers.reshape(
            x, shape=[batch_size, in_channels * 1 * self.filter_size])
        out = fluid.layers.matmul(inputs, self.temp_weight, transpose_y=True)
        out = fluid.layers.elementwise_add(out, self.conv._bias_param, axis=-1)
        out = fluid.layers.reshape(out, out.shape + [1, 1])
        out = self._helper.append_activation(out, act=self.act)
        return out

    def _reshaped_weight(self):
        """
        Get the linearized weight of convolution filter, cause it is by nature 
        a matmul weight. And because the model uses weight norm, compute the
        weight by weight_v * weight_g to make it faster.

        Returns:
            weight_matrix (Variable): Shape(C_out, C_in * 1 * kernel_size)
        """
        shape = self.conv._filter_param_v.shape
        matrix_shape = [shape[0], np.prod(shape[1:])]
        weight_matrix = fluid.layers.reshape(
            self.conv._filter_param_v, shape=matrix_shape)
        weight_matrix = fluid.layers.elementwise_mul(
            fluid.layers.l2_normalize(
                weight_matrix, axis=1),
            self.conv._filter_param_g,
            axis=0)
        return weight_matrix


class Conv1DTranspose(dg.Layer):
    """
    A convolutional transpose 1D block implemented with convolutional transpose
    2D. It does not ensure that the output is exactly expanded stride times in 
    time dimension.
    """

    def __init__(self,
                 name_scope,
                 in_channels,
                 num_filters,
                 filter_size,
                 padding=0,
                 stride=1,
                 dilation=1,
                 groups=None,
                 param_attr=None,
                 bias_attr=None,
                 use_cudnn=True,
                 act=None,
                 dtype="float32"):
        super(Conv1DTranspose, self).__init__(name_scope, dtype=dtype)

        self.in_channels = in_channels
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.groups = groups

        self.conv_transpose = Conv2DTranspose(
            self.full_name(),
            num_filters,
            filter_size=(1, filter_size),
            padding=(0, padding),
            stride=(1, stride),
            dilation=(1, dilation),
            groups=groups,
            param_attr=param_attr,
            bias_attr=bias_attr,
            use_cudnn=use_cudnn,
            act=act,
            dtype=dtype)

    def forward(self, x):
        """
        Argss:
            x (Variable): Shape(B, C_in, 1, T_in), where C_in means the input
                channels and T_in means the number of time steps of input.
        
        Returns:
            out (Variable): shape(B, C_out, 1, T_out), where C_out means the
                output channels and T_out means the number of time steps of
                input.
        """
        return self.conv_transpose(x)
