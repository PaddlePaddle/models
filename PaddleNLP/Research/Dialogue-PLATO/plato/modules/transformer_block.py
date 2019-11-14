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
TransformerBlock class.
"""

import paddle.fluid as fluid
from paddle.fluid.dygraph import FC
from paddle.fluid.dygraph import Layer
import paddle.fluid.layers as layers

from plato.modules.feedforward import FeedForward
from plato.modules.layer_norm import LayerNorm
from plato.modules.multihead_attention import MultiheadAttention
import plato.modules.functions as F


class TransformerBlock(Layer):
    """
    Transformer block module.
    """

    def __init__(self, name_scope, hidden_dim, num_heads, dropout, attn_dropout, ff_dropout):
        super().__init__(name_scope)

        self.attn = MultiheadAttention(name_scope=self.full_name(),
                                       hidden_dim=hidden_dim,
                                       num_heads=num_heads,
                                       dropout=attn_dropout)
        self.attn_norm = LayerNorm(name_scope=self.full_name(),
                                   begin_norm_axis=2,
                                   epsilon=1e-12,
                                   param_attr=fluid.ParamAttr(
                                       regularizer=fluid.regularizer.L2Decay(0.0)),
                                   bias_attr=fluid.ParamAttr(
                                       regularizer=fluid.regularizer.L2Decay(0.0)))
        self.ff = FeedForward(name_scope=self.full_name(),
                              hidden_dim=hidden_dim,
                              inner_dim=4 * hidden_dim,
                              dropout=ff_dropout)
        self.ff_norm = LayerNorm(name_scope=self.full_name(),
                                 begin_norm_axis=2,
                                 epsilon=1e-12,
                                 param_attr=fluid.ParamAttr(
                                     regularizer=fluid.regularizer.L2Decay(0.0)),
                                 bias_attr=fluid.ParamAttr(
                                     regularizer=fluid.regularizer.L2Decay(0.0)))
        self.dropout = dropout
        return

    def forward(self, inp, mask=None, cache=None):
        """
        Forward process on one transformer layer.

        @param : x
        @type : Variable(shape: [batch_size, seq_len, hidden_size])

        @param : memory
        @type : Variable(shape: [batch_size, seq_len, hidden_size])

        @param : mask

        @param : cache
        """
        attn_out = self.attn(inp, mask, cache)
        attn_out = F.dropout(attn_out, self.dropout)
        attn_out = self.attn_norm(attn_out + inp)

        ff_out = self.ff(attn_out)
        ff_out = F.dropout(ff_out, self.dropout)
        ff_out = self.ff_norm(ff_out + attn_out)

        return ff_out


def main():
    import numpy as np

    place = fluid.CPUPlace()
    with fluid.dygraph.guard(place):
        model = TransformerBlock("TransformerBlock", 10, 2, 0.5, 0.5, 0.5)
        inp = np.random.rand(2, 3, 10).astype("float32")
        inp = fluid.dygraph.to_variable(inp)
        out = model(inp, inp)
        print(out)


if __name__ == "__main__":
    main()
