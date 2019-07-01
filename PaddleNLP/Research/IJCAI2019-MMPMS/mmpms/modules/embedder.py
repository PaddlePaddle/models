#!/usr/bin/env python
# -*- coding: utf-8 -*-
################################################################################
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
################################################################################

from __future__ import division

import numpy as np
import paddle.fluid as fluid

import mmpms.layers as layers


class Embedder(layers.Embedding):
    def __init__(self,
                 num_embeddings,
                 embedding_dim,
                 is_sparse=False,
                 is_distributed=False,
                 padding_idx=None,
                 param_attr=None,
                 dtype='float32',
                 name=None):
        super(Embedder, self).__init__(
            size=[num_embeddings, embedding_dim],
            is_sparse=is_sparse,
            is_distributed=is_distributed,
            padding_idx=padding_idx,
            param_attr=param_attr,
            dtype=dtype,
            name=name)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

    def from_pretrained(self, embeds, place, scale=0.05):
        assert len(embeds) == self.num_embeddings
        assert len(embeds[0]) == self.embedding_dim

        embeds = np.array(embeds, dtype='float32')
        num_known = 0
        for i in range(len(embeds)):
            if np.all(embeds[i] == 0):
                embeds[i] = np.random.uniform(
                    low=-scale, high=scale, size=self.embedding_dim)
            else:
                num_known += 1
        if self.padding_idx is not None:
            embeds[self.padding_idx] = 0

        embedding_param = fluid.global_scope().find_var(
            self.param_attr.name).get_tensor()
        embedding_param.set(embeds, place)

        print("{} words have pretrained embeddings ".format(num_known) +
              "(coverage: {:.3f})".format(num_known / self.num_embeddings))
