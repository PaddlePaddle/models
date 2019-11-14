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
MultiheadAttention class.
"""

import paddle.fluid as fluid
from paddle.fluid.dygraph import Layer
from paddle.fluid.dygraph import FC
import paddle.fluid.layers as layers

import plato.modules.functions as F


class MultiheadAttention(Layer):
    """
    Multi head attention layer.
    """

    def __init__(self, name_scope, hidden_dim, num_heads, dropout):
        assert hidden_dim % num_heads == 0
        super().__init__(name_scope)

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.linear_qkv = FC(name_scope=self.full_name(),
                             size=hidden_dim * 3,
                             num_flatten_dims=2)
        self.linear_out = FC(name_scope=self.full_name(),
                             size=hidden_dim,
                             num_flatten_dims=2)
        self.dropout = dropout
        return

    def _split_heads(self, x, is_key=False):
        x = layers.reshape(
            x=x, shape=[0, 0, self.num_heads, self.head_dim]
        )
        x = layers.transpose(x=x, perm=[0, 2, 3, 1] if is_key else [0, 2, 1, 3])
        return x

    def _merge_heads(self, x):
        x = layers.transpose(x=x, perm=[0, 2, 1, 3])
        x = layers.reshape(x=x, shape=[0, 0, self.hidden_dim])
        return x

    def _attn(self, query, key, value, mask):
        # shape: [batch_size, num_head, seq_len, seq_len]
        scores = layers.matmul(x=query, y=key, alpha=self.scale)

        if mask is not None:
            mask = F.unsqueeze(mask, [1])
            mask = layers.expand(mask, [1, self.num_heads, 1, 1])
            mask.stop_gradient = True
            scores = (1 - mask) * scores + layers.scale(mask, scale=-1e10)

        attn = layers.softmax(scores, axis=-1)
        attn = F.dropout(attn, self.dropout)

        if mask is not None:
            attn = (1 - mask) * attn

        out = layers.matmul(x=attn, y=value)
        return out

    def forward(self, inp, mask=None, cache=None):
        """ Forward process of self attention. """
        # shape: [batch_size, seq_len, 3 * hidden_dim]
        qkv = self.linear_qkv(inp)
        query, key, value = layers.split(qkv, num_or_sections=3, dim=2)


        # shape: [batch_size, num_head, seq_len, head_dim]
        query = self._split_heads(query)
        # shape: [batch_size, num_head, head_dim, seq_len]
        key = self._split_heads(key, is_key=True)
        # shape: [batch_size, num_head, seq_len, head_dim]
        value = self._split_heads(value)

        if cache is not None:
            if "key" in cache and "value" in cache:
                key = layers.concat([cache["key"], key], axis=3)
                value = layers.concat([cache["value"], value], axis=2)
            cache["key"] = key
            cache["value"] = value

        out = self._attn(query, key, value, mask)
        out = self._merge_heads(out)
        out = self.linear_out(out)
        return out


def main():
    import numpy as np

    place = fluid.CPUPlace()
    with fluid.dygraph.guard(place):
        model = MultiheadAttention("MultiheadAttention", 10, 2, 0.5)
        inp = np.random.rand(2, 3, 10).astype("float32")
        inp = fluid.dygraph.to_variable(inp)
        out = model(inp, inp, inp)
        print(out)


if __name__ == "__main__":
    main()
