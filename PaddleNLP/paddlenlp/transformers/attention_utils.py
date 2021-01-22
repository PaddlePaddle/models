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

import numpy as np
import copy

from paddle import ParamAttr
from paddle.nn import Layer
import paddle
import collections
from paddle.nn import Linear, Dropout, LayerNorm, LayerList, Layer
import paddle.nn as nn
import paddle.nn.functional as F


class Registry(object):
    def __init__(self):
        self.cls_dict = {}

    def register(self, name):
        def add_item(name, cls):
            self.cls_dict[name] = cls
            return cls

        return lambda cls: add_item(name, cls)


AttentionRegistry = Registry()


class Mask(object):
    def __init__(self,
                 query_length,
                 key_length,
                 num_heads,
                 block_size,
                 window_size,
                 num_global_blocks,
                 num_rand_blocks,
                 seed=None):
        for k, v in locals().items():
            if k != "self":
                setattr(self, k, v)
        self.mask = np.zeros_like(
            np.arange(query_length * key_length * num_heads).reshape((
                num_heads, query_length, key_length)))
        self.rand_mask = np.zeros_like(
            np.arange(query_length * key_length * num_heads).reshape((
                num_heads, query_length, key_length)))
        self.rand_mask_idx = [[] for i in range(num_heads)]
        self.num_query_blocks = self.query_length // self.block_size     \
                + int(self.query_length % self.block_size != 0)
        self.num_key_blocks = self.key_length // self.block_size         \
                + int(self.key_length % self.block_size != 0)
        self.num_window_blocks = self.window_size // 2
        if seed:
            np.random.seed(seed)
        # create global mask
        self._create_global_mask()
        # create window mask
        self._create_window_mask()
        # create random mask
        self._create_random_mask()

    def get_mask(self):
        return self.mask

    def get_rand_mask_idx(self):
        return self.rand_mask_idx

    def get_rand_mask(self):
        return self.rand_mask

    def get_float_mask(self):
        float_mask = np.array(self.mask, dtype='float32')
        float_mask[float_mask != 1] = -np.inf
        float_mask[float_mask == 1.] = 0
        return float_mask

    def _create_global_mask(self):
        global_block_length = self.num_global_blocks * self.block_size
        self.mask[:, 0:global_block_length, :] = 1
        self.mask[:, :, 0:global_block_length] = 1

    def _create_window_mask(self):
        for query_block_idx in range(self.num_query_blocks):
            left_key_block_idx, right_key_block_idx = self._get_window_block_idx(
                query_block_idx)
            left_idx = left_key_block_idx * self.block_size
            right_idx = (right_key_block_idx + 1) * self.block_size
            query_left_idx = query_block_idx * self.block_size
            query_right_idx = min((query_block_idx + 1) * self.block_size,
                                  self.query_length)
            self.mask[:, query_left_idx:query_right_idx, left_idx:right_idx] = 1

    def _create_random_mask(self):
        all_key_blocks_idx = np.arange(0, self.num_key_blocks, dtype=np.int32)
        for query_block_idx in range(self.num_query_blocks):
            left_key_block_idx, right_key_block_idx = self._get_window_block_idx(
                query_block_idx)
            illegal_blocks_idx = [
                i for i in range(left_key_block_idx, right_key_block_idx + 1)
            ]
            illegal_blocks_idx.extend(
                [i for i in range(self.num_global_blocks)])
            left_key_block_idx = query_block_idx - self.num_window_blocks
            right_key_block_idx = query_block_idx + self.num_window_blocks
            if self.num_global_blocks > left_key_block_idx:
                num_fill_blocks = self.num_global_blocks - left_key_block_idx
                illegal_blocks_idx.extend([
                    i
                    for i in range(self.num_key_blocks - num_fill_blocks,
                                   self.num_key_blocks)
                ])
            if right_key_block_idx >= self.num_key_blocks:
                num_fill_blocks = right_key_block_idx - self.num_key_blocks + 1
                illegal_blocks_idx.extend([
                    i
                    for i in range(self.num_global_blocks,
                                   self.num_global_blocks + num_fill_blocks)
                ])

            illegal_blocks_idx = set(illegal_blocks_idx)

            query_left_idx = query_block_idx * self.block_size
            query_right_idx = min((query_block_idx + 1) * self.block_size,
                                  self.query_length)
            for i in range(self.num_heads):
                legal_blocks_idx = []
                legal_idx = []
                perm_block = np.random.permutation(all_key_blocks_idx)
                for j in perm_block:
                    if j not in illegal_blocks_idx:
                        legal_blocks_idx.append(j)
                    if len(legal_blocks_idx) == self.num_rand_blocks:
                        break
                for j in legal_blocks_idx:
                    key_left_idx = j * self.block_size
                    key_right_idx = min((j + 1) * self.block_size,
                                        self.key_length)
                    legal_idx.extend(
                        [i for i in range(key_left_idx, key_right_idx)])
                    self.rand_mask[i, query_left_idx:query_right_idx,
                                   key_left_idx:key_right_idx] = 1
                self.rand_mask_idx[i].append(legal_blocks_idx)
        self.rand_mask_idx = np.stack(self.rand_mask_idx, axis=0)
        self.rand_mask_idx = self.rand_mask_idx[:, self.num_global_blocks:]
        self.mask = np.maximum(self.rand_mask, self.mask)

    def _get_window_block_idx(self, query_block_idx):
        left_key_block_idx = max(0, query_block_idx - self.num_window_blocks)
        right_key_block_idx = min(query_block_idx + self.num_window_blocks,
                                  self.num_key_blocks - 1)
        return left_key_block_idx, right_key_block_idx


def _convert_param_attr_to_list(param_attr, n):
    if isinstance(param_attr, (list, tuple)):
        assert len(param_attr) == n, (
            "length of param_attr should be %d when it is a list/tuple" % n)
        param_attrs = []
        for attr in param_attr:
            if isinstance(attr, bool):
                if attr:
                    param_attrs.append(ParamAttr._to_attr(None))
                else:
                    param_attrs.append(False)
            else:
                param_attrs.append(ParamAttr._to_attr(attr))
    elif isinstance(param_attr, bool):
        param_attrs = []
        if param_attr:
            param_attrs = [ParamAttr._to_attr(None) for i in range(n)]
        else:
            param_attrs = [False] * n
    else:
        param_attrs = []
        attr = ParamAttr._to_attr(param_attr)
        for i in range(n):
            attr_i = copy.deepcopy(attr)
            if attr.name:
                attr_i.name = attr_i.name + "_" + str(i)
            param_attrs.append(attr_i)
    return param_attrs


class Linear3D(Layer):
    def __init__(self,
                 hidden_size,
                 num_attention_heads,
                 size_per_head,
                 weight_attr=None,
                 bias_attr=None):
        super(Linear3D, self).__init__()
        self._dtype = self._helper.get_default_dtype()
        self._weight_attr = weight_attr
        self._bias_attr = bias_attr
        self.weight = self.create_parameter(
            shape=[hidden_size, hidden_size],
            attr=self._weight_attr,
            dtype=self._dtype,
            is_bias=False)
        self.bias = self.create_parameter(
            shape=[hidden_size],
            attr=self._bias_attr,
            dtype=self._dtype,
            is_bias=True)
        self.size_per_head = size_per_head
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size

    def forward(self, input):
        # abc,cde->adbe
        reshape_input = paddle.unsqueeze(input, 1)
        reshape_w = paddle.reshape(
            self.weight,
            [self.hidden_size, self.num_attention_heads, self.size_per_head])
        reshape_w = paddle.transpose(reshape_w, [1, 0, 2])
        reshape_w = paddle.unsqueeze(reshape_w, 0)
        result = paddle.matmul(reshape_input, reshape_w)
        reshape_b = paddle.reshape(
            self.bias, [1, self.num_attention_heads, 1, self.size_per_head])
        result += reshape_b
        return result


class LinearProj3D(Layer):
    def __init__(self,
                 hidden_size,
                 num_attention_heads,
                 size_per_head,
                 weight_attr=None,
                 bias_attr=None):
        super(LinearProj3D, self).__init__()
        self._dtype = self._helper.get_default_dtype()
        self._weight_attr = weight_attr
        self._bias_attr = bias_attr
        self.weight = self.create_parameter(
            shape=[hidden_size, hidden_size],
            attr=self._weight_attr,
            dtype=self._dtype,
            is_bias=False)
        self.bias = self.create_parameter(
            shape=[hidden_size],
            attr=self._bias_attr,
            dtype=self._dtype,
            is_bias=True)
        self.size_per_head = size_per_head
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size

    def forward(self, input):
        # BFNH,NHD->BFD
        result = paddle.matmul(input, self.weight)


class MultiHeadAttention(Layer):

    Cache = collections.namedtuple("Cache", ["k", "v"])
    StaticCache = collections.namedtuple("StaticCache", ["k", "v"])

    def __init__(self,
                 embed_dim,
                 num_heads,
                 dropout=0.,
                 kdim=None,
                 vdim=None,
                 weight_attr=None,
                 bias_attr=None,
                 block_size=1,
                 window_size=3,
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

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = Linear3D(
            embed_dim,
            num_heads,
            self.head_dim,
            weight_attr,
            bias_attr=bias_attr)
        self.k_proj = Linear3D(
            embed_dim,
            num_heads,
            self.head_dim,
            weight_attr,
            bias_attr=bias_attr)
        self.v_proj = Linear3D(
            embed_dim,
            num_heads,
            self.head_dim,
            weight_attr,
            bias_attr=bias_attr)
        self.out_proj = nn.Linear(
            embed_dim, embed_dim, weight_attr, bias_attr=bias_attr)

        self.attn_impl = AttentionRegistry.cls_dict[attention_type](
            num_heads, block_size, window_size, num_global_blocks,
            num_rand_blocks, seed)

    def _prepare_qkv(self, query, key, value, cache=None):
        q = self.q_proj(query)

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

        out = self.attn_impl(q, k, v, self.head_dim, attn_mask, self.dropout)

        # combine heads
        out = paddle.transpose(out, perm=[0, 2, 1, 3])
        out = paddle.reshape(x=out, shape=[0, 0, out.shape[2] * out.shape[3]])

        # project to output
        out = self.out_proj(out)

        outs = [out]
        if cache is not None:
            outs.append(cache)
        return out if len(outs) == 1 else tuple(outs)
