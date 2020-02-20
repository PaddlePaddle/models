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
MMDNN class
"""
import numpy as np
import paddle.fluid as fluid
import logging
from paddle.fluid.dygraph import Embedding, LayerNorm, Linear, to_variable, Layer, guard
from paddle.fluid.dygraph.nn import Conv2D

import paddle_layers as pd_layers

from paddle.fluid import layers
from paddle.fluid.dygraph import Layer

class BasicLSTMUnit(Layer):
    """
    ****
    BasicLSTMUnit class, Using basic operator to build LSTM
    The algorithm can be described as the code below.
        .. math::
           i_t &= \sigma(W_{ix}x_{t} + W_{ih}h_{t-1} + b_i)
           f_t &= \sigma(W_{fx}x_{t} + W_{fh}h_{t-1} + b_f + forget_bias )
           o_t &= \sigma(W_{ox}x_{t} + W_{oh}h_{t-1} + b_o)
           \\tilde{c_t} &= tanh(W_{cx}x_t + W_{ch}h_{t-1} + b_c)
           c_t &= f_t \odot c_{t-1} + i_t \odot \\tilde{c_t}
           h_t &= o_t \odot tanh(c_t)
        - $W$ terms denote weight matrices (e.g. $W_{ix}$ is the matrix
          of weights from the input gate to the input)
        - The b terms denote bias vectors ($bx_i$ and $bh_i$ are the input gate bias vector).
        - sigmoid is the logistic sigmoid function.
        - $i, f, o$ and $c$ are the input gate, forget gate, output gate,
          and cell activation vectors, respectively, all of which have the same size as
          the cell output activation vector $h$.
        - The :math:`\odot` is the element-wise product of the vectors.
        - :math:`tanh` is the activation functions.
        - :math:`\\tilde{c_t}` is also called candidate hidden state,
          which is computed based on the current input and the previous hidden state.
    Args:
        name_scope(string) : The name scope used to identify parameter and bias name
        hidden_size (integer): The hidden size used in the Unit.
        param_attr(ParamAttr|None): The parameter attribute for the learnable
            weight matrix. Note:
            If it is set to None or one attribute of ParamAttr, lstm_unit will
            create ParamAttr as param_attr. If the Initializer of the param_attr
            is not set, the parameter is initialized with Xavier. Default: None.
        bias_attr (ParamAttr|None): The parameter attribute for the bias
            of LSTM unit.
            If it is set to None or one attribute of ParamAttr, lstm_unit will 
            create ParamAttr as bias_attr. If the Initializer of the bias_attr
            is not set, the bias is initialized as zero. Default: None.
        gate_activation (function|None): The activation function for gates (actGate).
                                  Default: 'fluid.layers.sigmoid'
        activation (function|None): The activation function for cells (actNode).
                             Default: 'fluid.layers.tanh'
        forget_bias(float|1.0): forget bias used when computing forget gate
        dtype(string): data type used in this unit
    """

    def __init__(self,
                 hidden_size,
                 input_size,
                 param_attr=None,
                 bias_attr=None,
                 gate_activation=None,
                 activation=None,
                 forget_bias=1.0,
                 dtype='float32'):
        super(BasicLSTMUnit, self).__init__(dtype)

        self._hiden_size = hidden_size
        self._param_attr = param_attr
        self._bias_attr = bias_attr
        self._gate_activation = gate_activation or layers.sigmoid
        self._activation = activation or layers.tanh
        self._forget_bias = layers.fill_constant(
            [1], dtype=dtype, value=forget_bias)
        self._forget_bias.stop_gradient = False
        self._dtype = dtype
        self._input_size = input_size

        self._weight = self.create_parameter(
            attr=self._param_attr,
            shape=[self._input_size + self._hiden_size, 4 * self._hiden_size],
            dtype=self._dtype)

        self._bias = self.create_parameter(
            attr=self._bias_attr,
            shape=[4 * self._hiden_size],
            dtype=self._dtype,
            is_bias=True)

    def forward(self, input, pre_hidden, pre_cell):
        concat_input_hidden = layers.concat([input, pre_hidden], 1)
        gate_input = layers.matmul(x=concat_input_hidden, y=self._weight)

        gate_input = layers.elementwise_add(gate_input, self._bias)
        i, j, f, o = layers.split(gate_input, num_or_sections=4, dim=-1)
        new_cell = layers.elementwise_add(
            layers.elementwise_mul(
                pre_cell,
                layers.sigmoid(layers.elementwise_add(f, self._forget_bias))),
            layers.elementwise_mul(layers.sigmoid(i), layers.tanh(j)))
        new_hidden = layers.tanh(new_cell) * layers.sigmoid(o)

        return new_hidden, new_cell


class MMDNN(object):
    """
    MMDNN
    """

    def __init__(self, config):
        """
        initialize
        """
        self.vocab_size = int(config['dict_size'])
        self.emb_size = int(config['net']['embedding_dim'])
        self.lstm_dim = int(config['net']['lstm_dim'])
        self.kernel_size = int(config['net']['num_filters'])
        self.win_size1 = int(config['net']['window_size_left'])
        self.win_size2 = int(config['net']['window_size_right'])
        self.dpool_size1 = int(config['net']['dpool_size_left'])
        self.dpool_size2 = int(config['net']['dpool_size_right'])
        self.hidden_size = int(config['net']['hidden_size'])
        self.seq_len1 = int(config['max_len_left'])
        self.seq_len2 = int(config['max_len_right'])
        self.task_mode = config['task_mode']

        if int(config['match_mask']) != 0:
            self.match_mask = True
        else:
            self.match_mask = False

        if self.task_mode == "pointwise":
            self.n_class = int(config['n_class'])
            self.out_size = self.n_class
        elif self.task_mode == "pairwise":
            self.out_size = 1
        else:
            logging.error("training mode not supported")

    def embedding_layer(self, input, zero_pad=True, scale=True):
        """
        embedding layer
        """
        emb = Embedding(
            size=[self.vocab_size, self.emb_size],
            padding_idx=(0 if zero_pad else None),
            param_attr=fluid.ParamAttr(
                name="word_embedding", initializer=fluid.initializer.Xavier()))
        emb = emb(input)
        if scale:
            emb = emb * (self.emb_size**0.5)
        return emb

    def bi_dynamic_lstm(self, input, hidden_size):
        """
        bi_lstm layer
        """
        fw_in_proj = Linear(
            input_dim=self.emb_size,
            output_dim=4 * hidden_size,
            param_attr=fluid.ParamAttr(name="fw_fc.w"),
            bias_attr=False)
        fw_in_proj = fw_in_proj(input)

        forward = pd_layers.DynamicLSTMLayer(
            size=4 * hidden_size,
            is_reverse=False,
            param_attr=fluid.ParamAttr(name="forward_lstm.w"),
            bias_attr=fluid.ParamAttr(name="forward_lstm.b")).ops()

        forward = forward(fw_in_proj)

        rv_in_proj = Linear(
            input_dim=self.emb_size,
            output_dim=4 * hidden_size,
            param_attr=fluid.ParamAttr(name="rv_fc.w"),
            bias_attr=False)
        rv_in_proj = rv_in_proj(input)

        reverse = pd_layers.DynamicLSTMLayer(
            4 * hidden_size,
            'lstm'
            is_reverse=True,
            param_attr=fluid.ParamAttr(name="reverse_lstm.w"),
            bias_attr=fluid.ParamAttr(name="reverse_lstm.b")).ops()
        reverse = reverse(rv_in_proj)

        return [forward, reverse]

    def conv_pool_relu_layer(self, input, mask=None):
        """
        convolution and pool layer
        """
        # data format NCHW
        emb_expanded = fluid.layers.unsqueeze(input=input, axes=[1])
        # same padding

        conv = Conv2d(
            num_filters=self.kernel_size,
            stride=1,
            padding=(int(self.seq_len1 / 2), int(self.seq_len2 // 2)),
            filter_size=(self.seq_len1, self.seq_len2),
            bias_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Constant(0.1)))
        conv = conv(emb_expanded)

        if mask is not None:
            cross_mask = fluid.layers.stack(x=[mask] * self.kernel_size, axis=1)
            conv = cross_mask * conv + (1 - cross_mask) * (-2**32 + 1)
        # valid padding
        pool = fluid.layers.pool2d(
            input=conv,
            pool_size=[
                int(self.seq_len1 / self.dpool_size1),
                int(self.seq_len2 / self.dpool_size2)
            ],
            pool_stride=[
                int(self.seq_len1 / self.dpool_size1),
                int(self.seq_len2 / self.dpool_size2)
            ],
            pool_type="max", )

        relu = fluid.layers.relu(pool)
        return relu

    def get_cross_mask(self, left_lens, right_lens):
        """
        cross mask
        """
        mask1 = fluid.layers.sequence_mask(
            x=left_lens, dtype='float32', maxlen=self.seq_len1 + 1)
        mask2 = fluid.layers.sequence_mask(
            x=right_lens, dtype='float32', maxlen=self.seq_len2 + 1)

        mask1 = fluid.layers.transpose(x=mask1, perm=[0, 2, 1])
        cross_mask = fluid.layers.matmul(x=mask1, y=mask2)
        return cross_mask

    def predict(self, left, right):
        """
        Forward network
        """
        left_emb = self.embedding_layer(left, zero_pad=True, scale=False)
        right_emb = self.embedding_layer(right, zero_pad=True, scale=False)

        bi_left_outputs = self.bi_dynamic_lstm(
            input=left_emb, hidden_size=self.lstm_dim)
        left_seq_encoder = fluid.layers.concat(input=bi_left_outputs, axis=1)
        bi_right_outputs = self.bi_dynamic_lstm(
            input=right_emb, hidden_size=self.lstm_dim)
        right_seq_encoder = fluid.layers.concat(input=bi_right_outputs, axis=1)

        pad_value = fluid.layers.assign(input=np.array([0]).astype("float32"))
        left_seq_encoder, left_lens = fluid.layers.sequence_pad(
            x=left_seq_encoder, pad_value=pad_value, maxlen=self.seq_len1)
        right_seq_encoder, right_lens = fluid.layers.sequence_pad(
            x=right_seq_encoder, pad_value=pad_value, maxlen=self.seq_len2)

        cross = fluid.layers.matmul(
            left_seq_encoder, right_seq_encoder, transpose_y=True)
        if self.match_mask:
            cross_mask = self.get_cross_mask(left_lens, right_lens)
        else:
            cross_mask = None

        conv_pool_relu = self.conv_pool_relu_layer(input=cross, mask=cross_mask)
        relu_hid1 = Linear(
            input_dim=conv_pool_relu.shape[-1],
            output_dim=self.hidden_size)
        relu_hid1 = relu_hid1(conv_pool_relu)
        relu_hid1 = fluid.layers.tanh(relu_hid1)

        relu_hid1 = Linear(
            input_dim=relu_hid1.shape[-1],
            output_dim=self.out_size)
        pred = relu_hid1(pred)

        pred = fluid.layers.softmax(pred)

        return left_seq_encoder, pred
