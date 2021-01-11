# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
from paddle.nn import Linear, Dropout, LayerNorm, LayerList, Layer
from paddle.nn import functional as F
from .attention import MultiHeadAttention
from .utils import _convert_param_attr_to_list


class TransformerDecoderLayer(Layer):
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward,
                 dropout=0.1,
                 activation="relu",
                 attn_dropout=None,
                 act_dropout=None,
                 normalize_before=False,
                 weight_attr=None,
                 bias_attr=None,
                 attention_type="default_attention"):
        self._config = locals()
        self._config.pop("self")
        self._config.pop("__class__", None)  # py3

        super(TransformerDecoderLayer, self).__init__()
        attn_dropout = dropout if attn_dropout is None else attn_dropout
        act_dropout = dropout if act_dropout is None else act_dropout
        self.normalize_before = normalize_before

        weight_attrs = _convert_param_attr_to_list(weight_attr, 3)
        bias_attrs = _convert_param_attr_to_list(bias_attr, 3)

        self.self_attn = MultiHeadAttention(
            d_model,
            nhead,
            dropout=attn_dropout,
            weight_attr=weight_attrs[0],
            bias_attr=bias_attrs[0],
            attention_type=attention_type)
        self.cross_attn = MultiHeadAttention(
            d_model,
            nhead,
            dropout=attn_dropout,
            weight_attr=weight_attrs[1],
            bias_attr=bias_attrs[1],
            attention_type=attention_type)
        self.linear1 = Linear(
            d_model, dim_feedforward, weight_attrs[2], bias_attr=bias_attrs[2])
        self.dropout = Dropout(act_dropout, mode="upscale_in_train")
        self.linear2 = Linear(
            dim_feedforward, d_model, weight_attrs[2], bias_attr=bias_attrs[2])
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout, mode="upscale_in_train")
        self.dropout2 = Dropout(dropout, mode="upscale_in_train")
        self.dropout3 = Dropout(dropout, mode="upscale_in_train")
        self.activation = getattr(F, activation)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, cache=None):
        residual = tgt
        if self.normalize_before:
            tgt = self.norm1(tgt)
        if cache is None:
            tgt = self.self_attn(tgt, tgt, tgt, tgt_mask, None)
        else:
            tgt, incremental_cache = self.self_attn(tgt, tgt, tgt, tgt_mask,
                                                    cache[0])
        tgt = residual + self.dropout1(tgt)
        if not self.normalize_before:
            tgt = self.norm1(tgt)

        residual = tgt
        if self.normalize_before:
            tgt = self.norm2(tgt)
        if cache is None:
            tgt = self.cross_attn(tgt, memory, memory, memory_mask, None)
        else:
            tgt, static_cache = self.cross_attn(tgt, memory, memory,
                                                memory_mask, cache[1])
        tgt = residual + self.dropout2(tgt)
        if not self.normalize_before:
            tgt = self.norm2(tgt)

        residual = tgt
        if self.normalize_before:
            tgt = self.norm3(tgt)
        tgt = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = residual + self.dropout3(tgt)
        if not self.normalize_before:
            tgt = self.norm3(tgt)
        return tgt if cache is None else (tgt, (incremental_cache,
                                                static_cache))

    def gen_cache(self, memory):
        incremental_cache = self.self_attn.gen_cache(
            memory, type=self.self_attn.Cache)
        static_cache = self.cross_attn.gen_cache(
            memory, memory, type=self.cross_attn.StaticCache)
        return incremental_cache, static_cache


class TransformerDecoder(Layer):
    def __init__(self, decoder_layer, num_layers, norm=None):
        super(TransformerDecoder, self).__init__()
        self.layers = LayerList([(decoder_layer if i == 0 else
                                  type(decoder_layer)(**decoder_layer._config))
                                 for i in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, cache=None):
        output = tgt
        new_caches = []
        for i, mod in enumerate(self.layers):
            if cache is None:
                output = mod(output,
                             memory,
                             tgt_mask=tgt_mask,
                             memory_mask=memory_mask,
                             cache=None)
            else:
                output, new_cache = mod(output,
                                        memory,
                                        tgt_mask=tgt_mask,
                                        memory_mask=memory_mask,
                                        cache=cache[i])
                new_caches.append(new_cache)

        if self.norm is not None:
            output = self.norm(output)

        return output if cache is None else (output, new_caches)

    def gen_cache(self, memory, do_zip=False):
        cache = [layer.gen_cache(memory) for layer in self.layers]
        if do_zip:
            cache = list(zip(*cache))
        return cache
