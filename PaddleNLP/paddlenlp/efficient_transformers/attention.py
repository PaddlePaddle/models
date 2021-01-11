#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import copy
import collections
import numpy as np

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn import Linear, Dropout, LayerNorm, LayerList, Layer
from ..utils.log import logger
from .registry import AttentionRegistry
from .masking import Mask


class Attention(Layer):
    def __init__(self,
                 num_heads=1,
                 block_size=1,
                 window_size=1,
                 num_global_blocks=1,
                 num_rand_blocks=1,
                 seed=None):
        super().__init__()

    def forward(self,
                query_matrix,
                key_matrix,
                value_matrix,
                d_head,
                attn_mask=None,
                dropout=None):
        raise NotImplementedError


@AttentionRegistry.register("default_attention")
class DefaultAttention(Attention):
    def forward(self,
                query_matrix,
                key_matrix,
                value_matrix,
                d_head,
                attn_mask=None,
                dropout=None):
        # scale dot product attention
        product = paddle.matmul(x=query_matrix, y=key_matrix, transpose_y=True)
        product = product * (d_head**-0.5)
        if attn_mask is not None:
            product = product + attn_mask
        weights = F.softmax(product)
        if dropout:
            weights = F.dropout(
                weights,
                dropout,
                training=self.training,
                mode="upscale_in_train")

        out = paddle.matmul(weights, value_matrix)
        return out, weights


@AttentionRegistry.register("bigbird_simulated")
class BigBirdSimulatedAttention(Attention):
    def __init__(self,
                 num_heads=1,
                 block_size=1,
                 window_size=1,
                 num_global_blocks=1,
                 num_rand_blocks=1,
                 seed=None):
        super(BigBirdSimulatedAttention,
              self).__init__(num_heads, block_size, window_size,
                             num_global_blocks, num_rand_blocks, seed)
        for k, v in locals().items():
            if k != "self":
                setattr(self, k, v)
        self.attn_impl = DefaultAttention(num_heads, block_size, window_size,
                                          num_global_blocks, num_rand_blocks,
                                          seed)

    def forward(self,
                query_matrix,
                key_matrix,
                value_matrix,
                d_head,
                attn_mask=None,
                dropout=None):
        query_length = query_matrix.shape[2]
        key_length = key_matrix.shape[2]
        # bool matrix
        mask = Mask(query_length, key_length, self.num_heads, self.block_size,
                    self.window_size, self.num_global_blocks,
                    self.num_rand_blocks, self.seed)
        mask = paddle.to_tensor(
            mask.get_float_mask(), dtype=paddle.get_default_dtype())
        if attn_mask is None:
            attn_mask = mask
        else:
            attn_mask = attn_mask + mask
        return self.attn_impl(
            query_matrix,
            key_matrix,
            value_matrix,
            d_head,
            attn_mask=attn_mask,
            dropout=dropout)


@AttentionRegistry.register("bigbird")
class BigBirdSparseAttention(Attention):
    def __init__(self):
        self.window_size = 0
        self.num_random_block = 0
        self.num_global_block = 0

    def forward(self,
                query_matrix,
                key_matrix,
                value_matrix,
                d_head,
                attn_mask=None,
                dropout=None):
        '''
            query_matrix: [B, H, T, D]
            key_matrix: [B, H, T, D]
            value_matrix: [B, H, T, D]
            
            Global Attention
            Random Attention
            Window Attention
            key_matrix分为五块：
            
        '''
        #
        #
        # 
        raise NotImplementedError


class MultiHeadAttention(Layer):

    Cache = collections.namedtuple("Cache", ["k", "v"])
    StaticCache = collections.namedtuple("StaticCache", ["k", "v"])

    def __init__(self,
                 embed_dim,
                 num_heads,
                 dropout=0.,
                 kdim=None,
                 vdim=None,
                 need_weights=False,
                 weight_attr=None,
                 bias_attr=None,
                 block_size=1,
                 window_size=1,
                 num_global_blocks=1,
                 num_rand_blocks=1,
                 seed=None,
                 attention_type="default_attention"):

        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.need_weights = need_weights

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(
            embed_dim, embed_dim, weight_attr, bias_attr=bias_attr)
        self.k_proj = nn.Linear(
            self.kdim, embed_dim, weight_attr, bias_attr=bias_attr)
        self.v_proj = nn.Linear(
            self.vdim, embed_dim, weight_attr, bias_attr=bias_attr)
        self.out_proj = nn.Linear(
            embed_dim, embed_dim, weight_attr, bias_attr=bias_attr)

        self.attn_impl = AttentionRegistry.cls_dict[attention_type](
            num_heads, block_size, window_size, num_global_blocks,
            num_rand_blocks, seed)

    def _prepare_qkv(self, query, key, value, cache=None):
        q = self.q_proj(query)
        q = paddle.reshape(x=q, shape=[0, 0, self.num_heads, self.head_dim])
        q = paddle.transpose(x=q, perm=[0, 2, 1, 3])

        if isinstance(cache, self.StaticCache):
            # for encoder-decoder attention in inference and has cached
            k, v = cache.k, cache.v
        else:
            k, v = self.compute_kv(key, value)

        if isinstance(cache, self.Cache):
            # for decoder self-attention in inference
            k = paddle.concat([cache.k, k], axis=2)
            v = paddle.concat([cache.v, v], axis=2)
            cache = self.Cache(k, v)

        return (q, k, v) if cache is None else (q, k, v, cache)

    def compute_kv(self, key, value):
        k = self.k_proj(key)
        v = self.v_proj(value)
        k = paddle.reshape(x=k, shape=[0, 0, self.num_heads, self.head_dim])
        k = paddle.transpose(x=k, perm=[0, 2, 1, 3])
        v = paddle.reshape(x=v, shape=[0, 0, self.num_heads, self.head_dim])
        v = paddle.transpose(x=v, perm=[0, 2, 1, 3])
        return k, v

    def gen_cache(self, key, value=None, type=Cache):
        if type == MultiHeadAttention.StaticCache:  # static_kv
            k, v = self.compute_kv(key, value)
            return self.StaticCache(k, v)
        elif value is None:  # incremental_state
            k = paddle.full(
                shape=[-1, self.num_heads, 0, self.head_dim],
                fill_value=0,
                dtype=key.dtype)

            v = paddle.full(
                shape=[-1, self.num_heads, 0, self.head_dim],
                fill_value=0,
                dtype=key.dtype)
            return self.Cache(k, v)
        else:
            # incremental_state with initial value, mainly for usage like UniLM
            return self.Cache(key, value)

    def forward(self, query, key, value, attn_mask=None, cache=None):
        key = query if key is None else key
        value = query if value is None else value
        # compute q ,k ,v
        if cache is None:
            q, k, v = self._prepare_qkv(query, key, value, cache)
        else:
            q, k, v, cache = self._prepare_qkv(query, key, value, cache)

        out, weights = self.attn_impl(q, k, v, self.head_dim, attn_mask,
                                      self.dropout)

        # combine heads
        out = paddle.transpose(out, perm=[0, 2, 1, 3])
        out = paddle.reshape(x=out, shape=[0, 0, out.shape[2] * out.shape[3]])

        # project to output
        out = self.out_proj(out)

        outs = [out]
        if self.need_weights:
            outs.append(weights)
        if cache is not None:
            outs.append(cache)
        return out if len(outs) == 1 else tuple(outs)
