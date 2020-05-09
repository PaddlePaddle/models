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
"dygraph transformer layers"

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph import Embedding, LayerNorm, Linear, Layer


class PrePostProcessLayer(Layer):
    """
    PrePostProcessLayer
    """

    def __init__(self, process_cmd, d_model, dropout_rate, name):
        super(PrePostProcessLayer, self).__init__()
        self.process_cmd = process_cmd
        self.functors = []
        self.exec_order = ""

        for cmd in self.process_cmd:
            if cmd == "a":  # add residual connection
                self.functors.append(lambda x, y: x + y if y is not None else x)
                self.exec_order += "a"
            elif cmd == "n":  # add layer normalization
                self.functors.append(
                    self.add_sublayer(
                        "layer_norm_%d" % len(
                            self.sublayers(include_sublayers=False)),
                        LayerNorm(
                            normalized_shape=d_model,
                            param_attr=fluid.ParamAttr(
                                name=name + "_layer_norm_scale",
                                initializer=fluid.initializer.Constant(1.)),
                            bias_attr=fluid.ParamAttr(
                                name=name + "_layer_norm_bias",
                                initializer=fluid.initializer.Constant(0.)))))
                self.exec_order += "n"
            elif cmd == "d":  # add dropout
                if dropout_rate:
                    self.functors.append(lambda x: fluid.layers.dropout(
                        x, dropout_prob=dropout_rate, is_test=False))
                    self.exec_order += "d"

    def forward(self, x, residual=None):
        for i, cmd in enumerate(self.exec_order):
            if cmd == "a":
                x = self.functors[i](x, residual)
            else:
                x = self.functors[i](x)
        return x


class PositionwiseFeedForwardLayer(Layer):
    """
    PositionwiseFeedForwardLayer
    """

    def __init__(self,
                 hidden_act,
                 d_inner_hid,
                 d_model,
                 dropout_rate,
                 param_initializer=None,
                 name=""):
        super(PositionwiseFeedForwardLayer, self).__init__()

        self._i2h = Linear(
            input_dim=d_model,
            output_dim=d_inner_hid,
            param_attr=fluid.ParamAttr(
                name=name + '_fc_0.w_0', initializer=param_initializer),
            bias_attr=name + '_fc_0.b_0',
            act=hidden_act)

        self._h2o = Linear(
            input_dim=d_inner_hid,
            output_dim=d_model,
            param_attr=fluid.ParamAttr(
                name=name + '_fc_1.w_0', initializer=param_initializer),
            bias_attr=name + '_fc_1.b_0')

        self._dropout_rate = dropout_rate

    def forward(self, x):
        """
        forward
        :param x:
        :return:
        """
        hidden = self._i2h(x)
        if self._dropout_rate:
            hidden = fluid.layers.dropout(
                hidden,
                dropout_prob=self._dropout_rate,
                upscale_in_train="upscale_in_train",
                is_test=False)
        out = self._h2o(hidden)
        return out


class MultiHeadAttentionLayer(Layer):
    """
    MultiHeadAttentionLayer
    """

    def __init__(self,
                 d_key,
                 d_value,
                 d_model,
                 n_head=1,
                 dropout_rate=0.,
                 cache=None,
                 gather_idx=None,
                 static_kv=False,
                 param_initializer=None,
                 name=""):
        super(MultiHeadAttentionLayer, self).__init__()
        self._n_head = n_head
        self._d_key = d_key
        self._d_value = d_value
        self._d_model = d_model
        self._dropout_rate = dropout_rate

        self._q_fc = Linear(
            input_dim=d_model,
            output_dim=d_key * n_head,
            param_attr=fluid.ParamAttr(
                name=name + '_query_fc.w_0', initializer=param_initializer),
            bias_attr=name + '_query_fc.b_0')

        self._k_fc = Linear(
            input_dim=d_model,
            output_dim=d_key * n_head,
            param_attr=fluid.ParamAttr(
                name=name + '_key_fc.w_0', initializer=param_initializer),
            bias_attr=name + '_key_fc.b_0')

        self._v_fc = Linear(
            input_dim=d_model,
            output_dim=d_value * n_head,
            param_attr=fluid.ParamAttr(
                name=name + '_value_fc.w_0', initializer=param_initializer),
            bias_attr=name + '_value_fc.b_0')

        self._proj_fc = Linear(
            input_dim=d_value * n_head,
            output_dim=d_model,
            param_attr=fluid.ParamAttr(
                name=name + '_output_fc.w_0', initializer=param_initializer),
            bias_attr=name + '_output_fc.b_0')

    def forward(self, queries, keys, values, attn_bias):
        """
        forward
        :param queries:
        :param keys:
        :param values:
        :param attn_bias:
        :return:
        """
        # compute q ,k ,v
        keys = queries if keys is None else keys
        values = keys if values is None else values

        q = self._q_fc(queries)
        k = self._k_fc(keys)
        v = self._v_fc(values)

        # split head

        q_hidden_size = q.shape[-1]
        reshaped_q = fluid.layers.reshape(
            x=q,
            shape=[0, 0, self._n_head, q_hidden_size // self._n_head],
            inplace=False)
        transpose_q = fluid.layers.transpose(x=reshaped_q, perm=[0, 2, 1, 3])

        k_hidden_size = k.shape[-1]
        reshaped_k = fluid.layers.reshape(
            x=k,
            shape=[0, 0, self._n_head, k_hidden_size // self._n_head],
            inplace=False)
        transpose_k = fluid.layers.transpose(x=reshaped_k, perm=[0, 2, 1, 3])

        v_hidden_size = v.shape[-1]
        reshaped_v = fluid.layers.reshape(
            x=v,
            shape=[0, 0, self._n_head, v_hidden_size // self._n_head],
            inplace=False)
        transpose_v = fluid.layers.transpose(x=reshaped_v, perm=[0, 2, 1, 3])

        scaled_q = fluid.layers.scale(x=transpose_q, scale=self._d_key**-0.5)
        # scale dot product attention
        product = fluid.layers.matmul(
            #x=transpose_q,
            x=scaled_q,
            y=transpose_k,
            transpose_y=True)
        #alpha=self._d_model**-0.5)
        if attn_bias is not None:
            product += attn_bias
        weights = fluid.layers.softmax(product)
        if self._dropout_rate:
            weights_droped = fluid.layers.dropout(
                weights,
                dropout_prob=self._dropout_rate,
                dropout_implementation="upscale_in_train",
                is_test=False)
            out = fluid.layers.matmul(weights_droped, transpose_v)
        else:
            out = fluid.layers.matmul(weights, transpose_v)

        # combine heads
        if len(out.shape) != 4:
            raise ValueError("Input(x) should be a 4-D Tensor.")
        trans_x = fluid.layers.transpose(out, perm=[0, 2, 1, 3])
        final_out = fluid.layers.reshape(
            x=trans_x,
            shape=[0, 0, trans_x.shape[2] * trans_x.shape[3]],
            inplace=False)

        # fc to output
        proj_out = self._proj_fc(final_out)
        return proj_out


class EncoderSubLayer(Layer):
    """
    EncoderSubLayer
    """

    def __init__(self,
                 hidden_act,
                 n_head,
                 d_key,
                 d_value,
                 d_model,
                 d_inner_hid,
                 prepostprocess_dropout,
                 attention_dropout,
                 relu_dropout,
                 preprocess_cmd="n",
                 postprocess_cmd="da",
                 param_initializer=None,
                 name=""):

        super(EncoderSubLayer, self).__init__()
        self.name = name
        self._preprocess_cmd = preprocess_cmd
        self._postprocess_cmd = postprocess_cmd
        self._prepostprocess_dropout = prepostprocess_dropout

        self._preprocess_layer = PrePostProcessLayer(
            self._preprocess_cmd,
            d_model,
            prepostprocess_dropout,
            name=name + "_pre_att")

        self._multihead_attention_layer = MultiHeadAttentionLayer(
            d_key,
            d_value,
            d_model,
            n_head,
            attention_dropout,
            None,
            None,
            False,
            param_initializer,
            name=name + "_multi_head_att")

        self._postprocess_layer = PrePostProcessLayer(
            self._postprocess_cmd,
            d_model,
            self._prepostprocess_dropout,
            name=name + "_post_att")
        self._preprocess_layer2 = PrePostProcessLayer(
            self._preprocess_cmd,
            d_model,
            self._prepostprocess_dropout,
            name=name + "_pre_ffn")

        self._positionwise_feed_forward = PositionwiseFeedForwardLayer(
            hidden_act,
            d_inner_hid,
            d_model,
            relu_dropout,
            param_initializer,
            name=name + "_ffn")

        self._postprocess_layer2 = PrePostProcessLayer(
            self._postprocess_cmd,
            d_model,
            self._prepostprocess_dropout,
            name=name + "_post_ffn")

    def forward(self, enc_input, attn_bias):
        """
        forward
        :param enc_input:
        :param attn_bias:
        :return:
        """
        pre_process_multihead = self._preprocess_layer(enc_input)

        attn_output = self._multihead_attention_layer(pre_process_multihead,
                                                      None, None, attn_bias)
        attn_output = self._postprocess_layer(attn_output, enc_input)

        pre_process2_output = self._preprocess_layer2(attn_output)

        ffd_output = self._positionwise_feed_forward(pre_process2_output)

        return self._postprocess_layer2(ffd_output, attn_output)


class EncoderLayer(Layer):
    """
    encoder
    """

    def __init__(self,
                 hidden_act,
                 n_layer,
                 n_head,
                 d_key,
                 d_value,
                 d_model,
                 d_inner_hid,
                 prepostprocess_dropout,
                 attention_dropout,
                 relu_dropout,
                 preprocess_cmd="n",
                 postprocess_cmd="da",
                 param_initializer=None,
                 name=""):

        super(EncoderLayer, self).__init__()
        self._preprocess_cmd = preprocess_cmd
        self._encoder_sublayers = list()
        self._prepostprocess_dropout = prepostprocess_dropout
        self._n_layer = n_layer
        self._hidden_act = hidden_act
        self._preprocess_layer = PrePostProcessLayer(
            self._preprocess_cmd, 3, self._prepostprocess_dropout,
            "post_encoder")

        for i in range(n_layer):
            self._encoder_sublayers.append(
                self.add_sublayer(
                    'esl_%d' % i,
                    EncoderSubLayer(
                        hidden_act,
                        n_head,
                        d_key,
                        d_value,
                        d_model,
                        d_inner_hid,
                        prepostprocess_dropout,
                        attention_dropout,
                        relu_dropout,
                        preprocess_cmd,
                        postprocess_cmd,
                        param_initializer,
                        name=name + '_layer_' + str(i))))

    def forward(self, enc_input, attn_bias):
        """
        forward
        :param enc_input:
        :param attn_bias:
        :return:
        """
        for i in range(self._n_layer):
            enc_output = self._encoder_sublayers[i](enc_input, attn_bias)
            enc_input = enc_output

        return self._preprocess_layer(enc_output)
