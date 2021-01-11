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
        self.query_length = query_length
        self.key_length = key_length
        self.block_size = block_size
        self.num_heads = num_heads
        self.window_size = window_size
        self.num_global_blocks = num_global_blocks
        self.num_rand_blocks = num_rand_blocks
        self.seed = seed
        self.mask = np.zeros_like(
            np.arange(query_length * key_length * num_heads).reshape((
                num_heads, query_length, key_length)))
        self.rand_mask = np.zeros_like(
            np.arange(query_length * key_length * num_heads).reshape((
                num_heads, query_length, key_length)))
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

    def get_rand_mask(self):
        return self.rand_mask

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
            illegal_blocks_idx = set(illegal_blocks_idx)
            query_left_idx = query_block_idx * self.block_size
            query_right_idx = min((query_block_idx + 1) * self.block_size,
                                  self.query_length)
            for i in range(self.num_heads):
                legal_blocks_idx = []
                perm_block = np.random.permutation(all_key_blocks_idx)
                for j in perm_block:
                    if j not in illegal_blocks_idx:
                        legal_blocks_idx.append(j)
                    if len(legal_blocks_idx) == self.num_rand_blocks:
                        break
                for j in legal_blocks_idx:
                    key_left_idx = j * self.block_size
                    key_right_idx = (j + 1) * self.block_size
                    self.rand_mask[i, query_left_idx:query_right_idx,
                                   key_left_idx:key_right_idx] = 1

        self.mask = np.maximum(self.rand_mask, self.mask)

    def _get_window_block_idx(self, query_block_idx):
        left_key_block_idx = max(0, query_block_idx - self.num_window_blocks)
        right_key_block_idx = min(query_block_idx + self.num_window_blocks,
                                  self.num_key_blocks - 1)
        return left_key_block_idx, right_key_block_idx


if __name__ == "__main__":
    mask = Mask(
        16,
        16,
        num_heads=2,
        block_size=2,
        window_size=3,
        num_global_blocks=1,
        num_rand_blocks=1)
    #print(mask.get_mask())
    print(mask.get_rand_mask())
