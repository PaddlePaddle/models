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
                 window_size=3,
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
        return out


@AttentionRegistry.register("bigbird_simulated")
class BigBirdSimulatedAttention(Attention):
    def __init__(self,
                 num_heads=1,
                 block_size=1,
                 window_size=3,
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
    def __init__(self,
                 num_heads=1,
                 block_size=1,
                 window_size=3,
                 num_global_blocks=1,
                 num_rand_blocks=1,
                 seed=None):
        super(BigBirdSparseAttention,
              self).__init__(num_heads, block_size, window_size,
                             num_global_blocks, num_rand_blocks, seed)
        for k, v in locals().items():
            if k != "self":
                setattr(self, k, v)

    def _get_random_key_value(self, key_matrix, value_matrix, attn_mask,
                              rand_mask_idx):
        rand_query_length = rand_mask_idx.shape[-2]
        rand_num = rand_mask_idx.shape[-1]
        batch_size = key_matrix.shape[0]
        gathered_key_list = []
        gathered_value_list = []
        rand_mask_list = []
        global_block_length = self.num_global_blocks * self.block_size
        for i in range(self.num_heads):
            rand_mask_idx_1d = paddle.reshape(rand_mask_idx[i], [-1])
            gathered_key_list.append(
                paddle.gather(
                    key_matrix[:, i], rand_mask_idx_1d, axis=1))
            gathered_value_list.append(
                paddle.gather(
                    value_matrix[:, i], rand_mask_idx_1d, axis=1))
            temp_rand_mask = [
                paddle.gather(attn_mask[i, j + global_block_length],
                              rand_mask_idx[i][j])
                for j in range(rand_query_length)
            ]
            rand_mask_list.append(paddle.stack(temp_rand_mask, axis=0))
        gathered_key = paddle.stack(gathered_key_list, axis=1)
        gathered_value = paddle.stack(gathered_value_list, axis=1)
        rand_mask = paddle.stack(rand_mask_list, axis=0)
        gathered_key = paddle.reshape(
            gathered_key,
            (batch_size, self.num_heads, rand_query_length, rand_num, -1))
        gathered_value = paddle.reshape(
            gathered_value,
            (batch_size, self.num_heads, rand_query_length, rand_num, -1))
        return gathered_key, gathered_value, rand_mask

    def _get_global_window_top_blocks(self, matrix, matrix_list,
                                      attn_mask=None):
        g = self.num_global_blocks
        w = self.window_size
        length = matrix.shape[2]
        for query_block_id in range(g, g + w // 2):
            left_block_id = query_block_id - w // 2
            right_block_id = query_block_id + w // 2
            right_block_length = min((right_block_id + 1) * self.block_size,
                                     length)
            # 需要填补的block数：g - left_key_block_id
            num_fill_blocks = g - left_block_id

            if attn_mask is None:
                block_list = [matrix[:, :, 0:right_block_length]]
                block_list.append(matrix[:, :, -num_fill_blocks *
                                         self.block_size:])
                block_list = paddle.concat(block_list, axis=2)
                matrix_list.append(paddle.unsqueeze(block_list, axis=2))
            else:
                block_list = [
                    attn_mask[:, query_block_id:query_block_id + 1, 0:
                              right_block_length]
                ]
                block_list.append(
                    attn_mask[:, query_block_id:query_block_id + 1,
                              -num_fill_blocks * self.block_size:])
                block_list = paddle.concat(block_list, axis=2)
                matrix_list.append(block_list)

    def _get_global_window_midlle_blocks(self,
                                         matrix,
                                         matrix_list,
                                         attn_mask=None):
        g = self.num_global_blocks
        w = self.window_size
        length = matrix.shape[2]
        global_block_length = self.num_global_blocks * self.block_size
        num_blocks = length // self.block_size \
                + int(length % self.block_size != 0)

        block_list = []
        for query_block_id in range(g + w // 2, g + w + w // 2):
            left_key_block_id = query_block_id - w // 2
            block_offset = num_blocks - w
            right_key_block_id = left_key_block_id + block_offset
            left_key_length = left_key_block_id * self.block_size
            right_key_length = right_key_block_id * self.block_size
            if attn_mask is None:
                block_list.append(
                    paddle.unsqueeze(
                        matrix[:, :, left_key_length:right_key_length], axis=2))
            else:
                block_list.append(
                    attn_mask[:, query_block_id:query_block_id + 1,
                              left_key_length:right_key_length])
        if attn_mask is None:
            block_list = paddle.concat(block_list, axis=2)
            block_list = paddle.transpose(block_list, [0, 1, 3, 2, 4])
            global_key = paddle.unsqueeze(
                matrix[:, :, 0:global_block_length], axis=2)
            global_key_shape = list(global_key.shape)
            global_key_shape[2] = block_list.shape[2]
            global_key = paddle.expand(global_key, global_key_shape)
            block_list = paddle.concat([block_list, global_key], axis=3)
        else:
            block_list = paddle.concat(block_list, axis=1)
            block_list = paddle.transpose(block_list, [0, 2, 1])
            start_attn_idx = global_block_length
            end_attn_idx = global_block_length + block_list.shape[
                1] * self.block_size
            global_key_mask = attn_mask[:, start_attn_idx:end_attn_idx, 0:
                                        global_block_length]
            block_list = paddle.concat([block_list, global_key_mask], axis=2)
        matrix_list.append(block_list)

    def _get_global_window_bottom_blocks(self,
                                         matrix,
                                         matrix_list,
                                         attn_mask=None):
        g = self.num_global_blocks
        w = self.window_size
        length = matrix.shape[2]
        global_block_length = self.num_global_blocks * self.block_size
        num_blocks = length // self.block_size \
                + int(length % self.block_size != 0)

        for query_block_id in range(num_blocks - w // 2, num_blocks):
            left_key_block_id = query_block_id - w // 2
            right_key_block_id = query_block_id + w // 2
            # 需要填补的block数
            num_fill_blocks = query_block_id - num_blocks + w // 2 + 1
            if attn_mask is None:
                block_list = [
                    matrix[:, :, 0:(num_fill_blocks + g) * self.block_size],
                    matrix[:, :, left_key_block_id * self.block_size:length]
                ]
                block_list = paddle.concat(block_list, axis=2)
                matrix_list.append(paddle.unsqueeze(block_list, axis=2))
            else:
                block_list = [
                    attn_mask[:, query_block_id:query_block_id + 1, 0:(
                        num_fill_blocks + g) * self.block_size],
                    attn_mask[:, query_block_id:query_block_id + 1,
                              left_key_block_id * self.block_size:length]
                ]
                block_list = paddle.concat(block_list, axis=2)
                matrix_list.append(block_list)

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
            key_matrix分为4块：
            
        '''
        query_length = query_matrix.shape[2]
        key_length = key_matrix.shape[2]
        batch_size = query_matrix.shape[0]

        num_query_blocks = query_length // self.block_size \
                + int(query_length % self.block_size != 0)
        num_key_blocks = key_length // self.block_size \
                + int(key_length % self.block_size != 0)

        # get mask
        mask = Mask(query_length, key_length, self.num_heads, self.block_size,
                    self.window_size, self.num_global_blocks,
                    self.num_rand_blocks, self.seed)
        rand_mask_idx = mask.get_rand_mask_idx()
        mask = paddle.to_tensor(
            mask.get_float_mask(), dtype=paddle.get_default_dtype())
        rand_mask_idx = paddle.to_tensor(rand_mask_idx, dtype='int32')
        if attn_mask is None:
            attn_mask = mask
        else:
            attn_mask = attn_mask + mask

        # gather random key,value
        random_keys, random_values, random_mask = self._get_random_key_value(
            key_matrix, value_matrix, attn_mask, rand_mask_idx)

        # global product [fix]
        # 所有global_block中的query与所有key做点积
        global_block_length = self.num_global_blocks * self.block_size
        global_product = paddle.matmul(
            query_matrix[:, :, 0:global_block_length],
            key_matrix,
            transpose_y=True)
        global_product = global_product * (d_head**-0.5)
        global_product = global_product + attn_mask[:, 0:global_block_length]
        global_weights = F.softmax(global_product)
        global_out = paddle.matmul(global_weights, value_matrix)
        global_out = paddle.unsqueeze(global_out, 2)

        # roll & product
        # 某些行中window_block数量较少，需要补齐
        # global_block + window_block + random_block
        second_key_matrix = []
        second_value_matrix = []
        second_mask_matrix = []
        mask_matrix = paddle.unsqueeze(attn_mask, axis=0)
        self._get_global_window_top_blocks(key_matrix, second_key_matrix)
        self._get_global_window_top_blocks(value_matrix, second_value_matrix)
        self._get_global_window_top_blocks(mask_matrix, second_mask_matrix,
                                           attn_mask)

        self._get_global_window_midlle_blocks(key_matrix, second_key_matrix)
        self._get_global_window_midlle_blocks(value_matrix, second_value_matrix)
        self._get_global_window_midlle_blocks(mask_matrix, second_mask_matrix,
                                              attn_mask)

        self._get_global_window_bottom_blocks(key_matrix, second_key_matrix)
        self._get_global_window_bottom_blocks(value_matrix, second_value_matrix)
        self._get_global_window_bottom_blocks(mask_matrix, second_mask_matrix,
                                              attn_mask)

        second_key_matrix = paddle.concat(second_key_matrix, axis=2)
        second_key_matrix = paddle.concat(
            [random_keys, second_key_matrix], axis=3)

        second_value_matrix = paddle.concat(second_value_matrix, axis=2)
        second_value_matrix = paddle.concat(
            [random_values, second_value_matrix], axis=3)

        second_mask_matrix = paddle.concat(second_mask_matrix, axis=1)
        second_mask_matrix = paddle.concat(
            [random_mask, second_mask_matrix], axis=2)
        mask_shape = list(second_mask_matrix.shape)
        second_mask_matrix = paddle.reshape(second_mask_matrix, [
            mask_shape[0], mask_shape[1] // self.block_size, self.block_size, -1
        ])

        second_query_matrix = paddle.unsqueeze(
            query_matrix[:, :, global_block_length:], axis=3)
        second_query_blocks = num_query_blocks - self.num_global_blocks
        second_query_matrix = paddle.reshape(second_query_matrix, [
            batch_size, self.num_heads, second_query_blocks, self.block_size, -1
        ])

        second_product = paddle.matmul(
            second_query_matrix, second_key_matrix, transpose_y=True)
        second_product = second_product * (d_head**-0.5)
        second_product = second_mask_matrix + second_product
        second_weights = F.softmax(second_product)
        second_out = paddle.matmul(second_weights, second_value_matrix)

        out = paddle.concat([global_out, second_out], axis=2)
        out = paddle.reshape(out,
                             [batch_size, self.num_heads, query_length, -1])

        return out


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
