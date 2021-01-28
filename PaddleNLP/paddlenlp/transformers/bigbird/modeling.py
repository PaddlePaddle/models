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
import numpy as np
from paddle.nn import Linear, Dropout, LayerNorm, LayerList, Layer
import paddle.nn.functional as F
import paddle.nn as nn
from ..attention_utils import _convert_param_attr_to_list, MultiHeadAttention, AttentionRegistry
from .. import PretrainedModel, register_base_model


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


def create_bigbird_attention_mask_list(
        num_layers, query_length, key_length, num_heads, block_size,
        window_size, num_global_blocks, num_rand_blocks, seed):
    attn_mask_list = []
    rand_mask_idx_list = []
    for i in range(num_layers):
        mask = Mask(query_length, key_length, num_heads, block_size,
                    window_size, num_global_blocks, num_rand_blocks, seed)
        attn_mask = paddle.to_tensor(mask.get_float_mask())
        rand_mask_idx = paddle.to_tensor(mask.get_rand_mask_idx())
        attn_mask_list.append(attn_mask)
        rand_mask_idx_list.append(rand_mask_idx)
    return attn_mask_list, rand_mask_idx_list


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
                rand_mask_idx=None,
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
                rand_mask_idx=None,
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
                rand_mask_idx=None,
                dropout=None):
        query_length = query_matrix.shape[2]
        key_length = key_matrix.shape[2]
        # bool matrix
        # mask = Mask(query_length, key_length, self.num_heads, self.block_size,
        #             self.window_size, self.num_global_blocks,
        #             self.num_rand_blocks, self.seed)
        # mask = paddle.to_tensor(
        #     mask.get_float_mask(), dtype=paddle.get_default_dtype())
        # if attn_mask is None:
        #     attn_mask = mask
        # else:
        #     attn_mask = attn_mask + mask
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
        rand_query_blocks = rand_mask_idx.shape[-2]
        rand_num = rand_mask_idx.shape[-1]
        batch_size = key_matrix.shape[0]
        gathered_key_list = []
        gathered_value_list = []
        rand_mask_list = []
        global_block_length = self.num_global_blocks * self.block_size
        query_blocks = attn_mask.shape[1] // self.block_size
        key_blocks = key_matrix.shape[2] // self.block_size
        value_blocks = value_matrix.shape[2] // self.block_size
        reshape_key_matrix = paddle.reshape(
            key_matrix,
            [batch_size, self.num_heads, key_blocks, self.block_size, -1])
        reshape_value_matrix = paddle.reshape(
            value_matrix,
            [batch_size, self.num_heads, value_blocks, self.block_size, -1])
        reshape_attn_mask = paddle.reshape(attn_mask, [
            self.num_heads, attn_mask.shape[1], key_blocks, self.block_size
        ])

        for i in range(self.num_heads):
            rand_mask_idx_1d = paddle.reshape(rand_mask_idx[i], [-1])
            gathered_key_list.append(
                paddle.gather(
                    reshape_key_matrix[:, i], rand_mask_idx_1d, axis=1))
            gathered_value_list.append(
                paddle.gather(
                    reshape_value_matrix[:, i], rand_mask_idx_1d, axis=1))
            temp_rand_mask = [
                paddle.gather(
                    reshape_attn_mask[i, (
                        j + self.num_global_blocks) * self.block_size:(
                            j + self.num_global_blocks + 1) * self.block_size],
                    rand_mask_idx[i][j],
                    axis=1) for j in range(rand_query_blocks)
            ]
            temp_rand_mask = paddle.stack(temp_rand_mask, axis=0)
            temp_rand_mask = paddle.reshape(temp_rand_mask, [
                rand_query_blocks * self.block_size, rand_num * self.block_size
            ])
            rand_mask_list.append(temp_rand_mask)
        gathered_key = paddle.stack(gathered_key_list, axis=1)
        gathered_value = paddle.stack(gathered_value_list, axis=1)
        rand_mask = paddle.stack(rand_mask_list, axis=0)
        rand_mask = paddle.reshape(rand_mask, [
            self.num_heads, rand_query_blocks * self.block_size,
            rand_num * self.block_size
        ])
        gathered_key = paddle.reshape(
            gathered_key, (batch_size, self.num_heads, rand_query_blocks,
                           rand_num * self.block_size, -1))
        gathered_value = paddle.reshape(
            gathered_value, (batch_size, self.num_heads, rand_query_blocks,
                             rand_num * self.block_size, -1))
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
                query_block_start_length = query_block_id * self.block_size
                query_block_end_length = (query_block_id + 1) * self.block_size
                block_list = [
                    attn_mask[:, query_block_start_length:
                              query_block_end_length, 0:right_block_length]
                ]
                block_list.append(
                    attn_mask[:, query_block_start_length:
                              query_block_end_length, -num_fill_blocks *
                              self.block_size:])
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
        block_offset = num_blocks - w // 2 * 2 - g
        block_list = []
        if attn_mask is None:
            for query_block_id in range(g + w // 2, g + w + w // 2):
                left_key_block_id = query_block_id - w // 2
                right_key_block_id = left_key_block_id + block_offset
                left_key_length = left_key_block_id * self.block_size
                right_key_length = right_key_block_id * self.block_size
                block_list.append(
                    paddle.unsqueeze(
                        matrix[:, :, left_key_length:right_key_length], axis=2))
            block_list = paddle.concat(block_list, axis=2)
            block_shape = block_list.shape
            block_list = paddle.reshape(block_list, [
                block_shape[0], block_shape[1], block_shape[2] *
                self.block_size, block_shape[3] // self.block_size, -1
            ])
            block_list = paddle.transpose(block_list, [0, 1, 3, 2, 4])
            global_key = paddle.unsqueeze(
                matrix[:, :, 0:global_block_length], axis=2)
            global_key_shape = list(global_key.shape)
            global_key_shape[2] = block_list.shape[2]
            global_key = paddle.expand(global_key, global_key_shape)
            block_list = paddle.concat([block_list, global_key], axis=3)
        else:
            for query_block_id in range(g + w // 2, g + w // 2 + block_offset):
                left_key_block_id = query_block_id - w // 2
                right_key_block_id = query_block_id + w // 2 + 1
                left_key_length = left_key_block_id * self.block_size
                right_key_length = right_key_block_id * self.block_size
                query_block_start_length = query_block_id * self.block_size
                query_block_end_length = (query_block_id + 1) * self.block_size
                block_list.append(attn_mask[:, query_block_start_length:
                                            query_block_end_length,
                                            left_key_length:right_key_length])
            block_list = paddle.concat(block_list, axis=1)
            start_attn_idx = global_block_length
            end_attn_idx = global_block_length + block_list.shape[1]
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
                query_block_start_length = query_block_id * self.block_size
                query_block_end_length = (query_block_id + 1) * self.block_size
                block_list = [
                    attn_mask[:, query_block_start_length:
                              query_block_end_length, 0:(num_fill_blocks + g
                                                         ) * self.block_size],
                    attn_mask[:, query_block_start_length:
                              query_block_end_length, left_key_block_id *
                              self.block_size:length]
                ]
                block_list = paddle.concat(block_list, axis=2)
                matrix_list.append(block_list)

    def forward(self,
                query_matrix,
                key_matrix,
                value_matrix,
                d_head,
                attn_mask=None,
                rand_mask_idx=None,
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
        # mask = Mask(query_length, key_length, self.num_heads, self.block_size,
        #             self.window_size, self.num_global_blocks,
        #             self.num_rand_blocks, self.seed)
        # rand_mask_idx = mask.get_rand_mask_idx()
        # mask = paddle.to_tensor(
        #     mask.get_float_mask(), dtype=paddle.get_default_dtype())
        # rand_mask_idx = paddle.to_tensor(rand_mask_idx, dtype='int32')
        # if attn_mask is None:
        #     attn_mask = mask
        # else:
        #     attn_mask = attn_mask + mask

        # gather random key,value
        random_keys, random_values, random_mask = self._get_random_key_value(
            key_matrix, value_matrix, attn_mask, rand_mask_idx)
        # 所有global_block中的query与所有key做点积
        global_block_length = self.num_global_blocks * self.block_size
        global_product = paddle.matmul(
            query_matrix[:, :, 0:global_block_length],
            key_matrix,
            transpose_y=True)
        global_product = global_product * (d_head**-0.5)

        global_product = global_product + attn_mask[:, 0:global_block_length]
        global_weights = F.softmax(global_product)
        if dropout:
            global_weights = F.dropout(
                global_weights,
                dropout,
                training=self.training,
                mode="upscale_in_train")
        global_out = paddle.matmul(global_weights, value_matrix)

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

        second_key_shape = second_key_matrix.shape
        second_key_matrix = paddle.unsqueeze(second_key_matrix, axis=3)
        second_key_matrix = paddle.expand(second_key_matrix, [
            second_key_shape[0], second_key_shape[1], second_key_shape[2],
            self.block_size, second_key_shape[3], second_key_shape[4]
        ])
        second_key_matrix = paddle.reshape(second_key_matrix, [
            second_key_shape[0], second_key_shape[1], second_key_shape[2] *
            self.block_size, second_key_shape[3], second_key_shape[4]
        ])

        second_value_matrix = paddle.concat(second_value_matrix, axis=2)
        second_value_matrix = paddle.concat(
            [random_values, second_value_matrix], axis=3)

        second_value_shape = second_value_matrix.shape
        second_value_matrix = paddle.unsqueeze(second_value_matrix, axis=3)
        second_value_matrix = paddle.expand(second_value_matrix, [
            second_value_shape[0], second_value_shape[1], second_value_shape[2],
            self.block_size, second_value_shape[3], second_value_shape[4]
        ])
        second_value_matrix = paddle.reshape(second_value_matrix, [
            second_value_shape[0], second_value_shape[1], second_value_shape[2]
            * self.block_size, second_value_shape[3], second_value_shape[4]
        ])

        second_mask_matrix = paddle.concat(second_mask_matrix, axis=1)
        second_mask_matrix = paddle.concat(
            [random_mask, second_mask_matrix], axis=2)

        second_query_matrix = paddle.unsqueeze(
            query_matrix[:, :, global_block_length:], axis=3)

        second_product = paddle.matmul(
            second_query_matrix, second_key_matrix, transpose_y=True)
        second_product = second_product * (d_head**-0.5)
        second_mask_matrix = paddle.unsqueeze(second_mask_matrix, 2)
        second_product = second_mask_matrix + second_product
        second_weights = F.softmax(second_product)
        if dropout:
            second_weights = F.dropout(
                second_weights,
                dropout,
                training=self.training,
                mode="upscale_in_train")
        second_out = paddle.matmul(second_weights, second_value_matrix)
        second_out = paddle.squeeze(second_out, 3)
        out = paddle.concat([global_out, second_out], axis=2)
        return out


class TransformerEncoderLayer(Layer):
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
                 attention_type="default_attention",
                 block_size=1,
                 window_size=3,
                 num_global_blocks=1,
                 num_rand_blocks=1,
                 seed=None):
        self._config = locals()
        self._config.pop("self")
        self._config.pop("__class__", None)  # py3

        super(TransformerEncoderLayer, self).__init__()
        attn_dropout = dropout if attn_dropout is None else attn_dropout
        act_dropout = dropout if act_dropout is None else act_dropout
        self.normalize_before = normalize_before

        weight_attrs = _convert_param_attr_to_list(weight_attr, 2)
        bias_attrs = _convert_param_attr_to_list(bias_attr, 2)

        self.self_attn = MultiHeadAttention(
            d_model,
            nhead,
            dropout=attn_dropout,
            weight_attr=weight_attrs[0],
            bias_attr=bias_attrs[0],
            attention_type=attention_type,
            block_size=block_size,
            window_size=window_size,
            num_global_blocks=num_global_blocks,
            num_rand_blocks=num_rand_blocks,
            seed=seed)
        self.linear1 = Linear(
            d_model, dim_feedforward, weight_attrs[1], bias_attr=bias_attrs[1])
        self.dropout = Dropout(act_dropout, mode="upscale_in_train")
        self.linear2 = Linear(
            dim_feedforward, d_model, weight_attrs[1], bias_attr=bias_attrs[1])
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout, mode="upscale_in_train")
        self.dropout2 = Dropout(dropout, mode="upscale_in_train")
        self.activation = getattr(F, activation)
        self.d_model = d_model

    def forward(self, src, src_mask=None, rand_mask_idx=None):
        residual = src
        if self.normalize_before:
            src = self.norm1(src)
        # TODO(guosheng): Add cache for encoder for the usage like UniLM
        src = self.self_attn(src, src, src, src_mask, rand_mask_idx)
        src = residual + self.dropout1(src)
        if not self.normalize_before:
            src = self.norm1(src)
        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src)
        if not self.normalize_before:
            src = self.norm2(src)
        return src


class TransformerEncoder(Layer):
    def __init__(self, encoder_layer, num_layers):
        super(TransformerEncoder, self).__init__()
        self.layers = LayerList([(encoder_layer if i == 0 else
                                  type(encoder_layer)(**encoder_layer._config))
                                 for i in range(num_layers)])
        self.num_layers = num_layers
        self.norm = LayerNorm(self.layers[0].d_model)
        self.normalize_before = self.layers[0].normalize_before

    def forward(self, src, src_mask_list=None, rand_mask_idx_list=None):
        output = src
        if not self.normalize_before:
            output = self.norm(output)

        for i, mod in enumerate(self.layers):
            if src_mask_list is None:
                output = mod(output)
            else:
                output = mod(output, src_mask_list[i], rand_mask_idx_list[i])

        if self.normalize_before:
            output = self.norm(output)

        return output


class BigBirdPooler(Layer):
    """
    """

    def __init__(self, hidden_size):
        super(BigBirdPooler, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BigBirdEmbeddings(Layer):
    """
    Include embeddings from word, position and token_type embeddings
    """

    def __init__(self,
                 vocab_size,
                 hidden_size=768,
                 hidden_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=16):
        super(BigBirdEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings,
                                                hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        if position_ids is None:
            ones = paddle.ones_like(input_ids, dtype="int64")
            seq_length = paddle.cumsum(ones, axis=1)
            position_ids = seq_length - ones
            position_ids.stop_gradient = True
        if token_type_ids is None:
            token_type_ids = paddle.zeros_like(input_ids, dtype="int64")

        input_embedings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = input_embedings + position_embeddings + token_type_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class BigBirdPretrainedModel(PretrainedModel):
    """
    An abstract class for pretrained BigBird models. It provides BigBird related
    `model_config_file`, `resource_files_names`, `pretrained_resource_files_map`,
    `pretrained_init_configuration`, `base_model_prefix` for downloading and
    loading pretrained models. See `PretrainedModel` for more details.
    """

    model_config_file = "model_config.json"
    pretrained_init_configuration = {
        "bert-base-uncased": {
            "vocab_size": 30522,
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 512,
            "type_vocab_size": 2,
            "initializer_range": 0.02,
            "pad_token_id": 0,
        },
    }
    resource_files_names = {"model_state": "model_state.pdparams"}
    pretrained_resource_files_map = {
        "model_state": {
            "bert-base-uncased":
            "https://paddlenlp.bj.bcebos.com/models/transformers/bert-base-uncased.pdparams",
        }
    }
    base_model_prefix = "bigbird"

    def init_weights(self, layer):
        """ Initialization hook """
        if isinstance(layer, (nn.Linear, nn.Embedding)):
            # In the dygraph mode, use the `set_value` to reset the parameter directly,
            # and reset the `state_dict` to update parameter in static mode.
            if isinstance(layer.weight, paddle.Tensor):
                layer.weight.set_value(
                    paddle.tensor.normal(
                        mean=0.0,
                        std=self.initializer_range
                        if hasattr(self, "initializer_range") else
                        self.bert.config["initializer_range"],
                        shape=layer.weight.shape))
        elif isinstance(layer, nn.LayerNorm):
            layer._epsilon = 1e-12


@register_base_model
class BigBirdModel(BigBirdPretrainedModel):
    def __init__(self,
                 num_layers,
                 vocab_size,
                 nhead,
                 attn_dropout=0.1,
                 dim_feedforward=3072,
                 activation="gelu",
                 normalize_before=False,
                 attention_type="bigbird",
                 block_size=1,
                 window_size=3,
                 num_global_blocks=1,
                 num_rand_blocks=2,
                 seed=None,
                 pad_token_id=0,
                 hidden_size=768,
                 hidden_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2):
        super(BigBirdModel, self).__init__()
        # embedding
        self.embeddings = BigBirdEmbeddings(
            vocab_size, hidden_size, hidden_dropout_prob,
            max_position_embeddings, type_vocab_size)

        # encoder
        encoder_layer = TransformerEncoderLayer(
            hidden_size,
            nhead,
            dim_feedforward,
            attn_dropout,
            activation,
            normalize_before=normalize_before,
            attention_type=attention_type,
            block_size=block_size,
            window_size=window_size,
            num_global_blocks=num_global_blocks,
            num_rand_blocks=num_rand_blocks,
            seed=seed)
        self.encoder = TransformerEncoder(encoder_layer, num_layers)
        # pooler
        self.pooler = nn.Linear(hidden_size, hidden_size)
        self.pad_token_id = pad_token_id

    def _process_mask(self, input_ids, attention_mask_list=None):
        #TODO: need to optimize mask processing
        attention_mask = paddle.unsqueeze(
            (input_ids == self.pad_token_id).astype(self.pooler.weight.dtype),
            axis=[1, 2])
        attention_mask = paddle.matmul(
            attention_mask, attention_mask, transpose_x=True) * -1e9
        if attention_mask_list is not None:
            for i in range(len(attention_mask_list)):
                attention_mask_list[i] = paddle.unsqueeze(
                    attention_mask_list[i], axis=0)
                attention_mask_list[i] += attention_mask

    def forward(self,
                input_ids,
                token_type_ids=None,
                attention_mask_list=None,
                rand_mask_idx_list=None):
        embedding_output = self.embeddings(input_ids, token_type_ids)
        self._process_mask(input_ids, attention_mask_list)
        encoder_output = self.encoder(embedding_output, attention_mask_list,
                                      rand_mask_idx_list)
        first_token_output = encoder_output[:, 0, :]
        pooled_output = paddle.tanh(self.pooler(first_token_output))
        return encoder_output, pooled_output


class BigBirdLMPredictionHead(Layer):
    def __init__(self,
                 hidden_size,
                 vocab_size,
                 activation,
                 embedding_weights=None):
        super(BigBirdLMPredictionHead, self).__init__()
        self.transform = nn.Linear(hidden_size, hidden_size)
        self.activation = getattr(nn.functional, activation)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.decoder_weight = self.create_parameter(
            shape=[hidden_size, vocab_size],
            dtype=self.transform.weight.dtype,
            is_bias=True) if embedding_weights is None else embedding_weights
        self.decoder_bias = self.create_parameter(
            shape=[vocab_size], dtype=self.decoder_weight.dtype, is_bias=True)

    def forward(self, hidden_states, masked_positions=None):
        if masked_positions is not None:
            hidden_states = paddle.reshape(hidden_states,
                                           [-1, hidden_states.shape[-1]])
            hidden_states = paddle.tensor.gather(hidden_states,
                                                 masked_positions)
        # gather masked tokens might be more quick
        hidden_states = self.transform(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = paddle.tensor.matmul(
            hidden_states, self.decoder_weight,
            transpose_y=True) + self.decoder_bias
        return hidden_states


class BigBirdPretrainingHeads(Layer):
    def __init__(self,
                 hidden_size,
                 vocab_size,
                 activation,
                 embedding_weights=None):
        super(BigBirdPretrainingHeads, self).__init__()
        self.predictions = BigBirdLMPredictionHead(
            hidden_size, vocab_size, activation, embedding_weights)
        self.seq_relationship = nn.Linear(hidden_size, 2)

    def forward(self, sequence_output, pooled_output, masked_positions=None):
        prediction_scores = self.predictions(sequence_output, masked_positions)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class BigBirdForPretraining(BigBirdPretrainedModel):
    def __init__(self, bert):
        super(BigBirdForPretraining, self).__init__()
        self.bert = bert
        self.cls = BigBirdPretrainingHeads(
            self.bert.config["hidden_size"],
            self.bert.config["vocab_size"],
            self.bert.config["hidden_act"],
            embedding_weights=self.bert.embeddings.word_embeddings.weight)

        self.apply(self.init_weights)

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None,
                masked_positions=None):
        outputs = self.bert(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask)
        sequence_output, pooled_output = outputs[:2]
        prediction_scores, seq_relationship_score = self.cls(
            sequence_output, pooled_output, masked_positions)
        return prediction_scores, seq_relationship_score


class BigBirdPretrainingCriterion(paddle.nn.Layer):
    def __init__(self, vocab_size):
        super(BigBirdPretrainingCriterion, self).__init__()
        # CrossEntropyLoss is expensive since the inner reshape (copy)
        self.loss_fn = paddle.nn.loss.CrossEntropyLoss(ignore_index=-1)
        self.vocab_size = vocab_size

    def forward(self, prediction_scores, seq_relationship_score,
                masked_lm_labels, next_sentence_labels, masked_lm_scale):
        masked_lm_loss = paddle.nn.functional.softmax_with_cross_entropy(
            prediction_scores, masked_lm_labels, ignore_index=-1)
        masked_lm_loss = masked_lm_loss / masked_lm_scale
        next_sentence_loss = paddle.nn.functional.softmax_with_cross_entropy(
            seq_relationship_score, next_sentence_labels)
        return paddle.sum(masked_lm_loss) + paddle.mean(next_sentence_loss)
