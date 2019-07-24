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

import mmpms.layers as layers


class Attention(object):
    def __init__(self, mode="mlp", memory_dim=None, hidden_dim=None, name=None):
        assert (mode in ["dot", "general", "mlp"]), (
            "Unsupported attention mode: {}".format(mode))
        self.name = name or "Attention"
        self.mode = mode

        if mode == "general":
            self.query_fc = layers.FC(size=memory_dim,
                                      bias_attr=False,
                                      name="{}.query".format(self.name))
            self.memory_dim = memory_dim
        elif mode == "mlp":
            assert hidden_dim is not None
            self.query_fc = layers.FC(size=hidden_dim,
                                      bias_attr=False,
                                      name="{}.query".format(self.name))
            self.memory_fc = layers.FC(size=hidden_dim,
                                       name="{}.memory".format(self.name))
            self.out_fc = layers.FC(size=1,
                                    bias_attr=False,
                                    name="{}.out".format(self.name))

    def __call__(self, query, memory, memory_proj=None):
        if self.mode == "dot":
            assert query.shape[-1] == memory.shape[-1]
            query_expand = layers.sequence_expand_as(x=query, y=memory)
            attn = layers.reduce_sum(
                layers.elementwise_mul(
                    x=query_expand, y=memory),
                dim=-1,
                keep_dim=True)
        elif self.mode == "general":
            assert self.memory_dim == memory.shape[-1]
            query_proj = self.query_fc(query)
            query_proj_expand = layers.sequence_expand_as(
                x=query_proj, y=memory)
            attn = layers.reduce_sum(
                layers.elementwise_mul(
                    x=query_proj_expand, y=memory),
                dim=-1,
                keep_dim=True)
        else:
            if memory_proj is None:
                memory_proj = self.memory_fc(memory)
            query_proj = self.query_fc(query)
            query_proj_expand = layers.sequence_expand_as(
                x=query_proj, y=memory_proj)
            hidden = layers.tanh(query_proj_expand + memory_proj)
            attn = self.out_fc(hidden)

        weights = layers.sequence_softmax(input=attn, use_cudnn=False)

        weights_reshape = layers.reshape(x=weights, shape=[-1])
        scaled = layers.elementwise_mul(x=memory, y=weights_reshape, axis=0)
        weighted_memory = layers.sequence_pool(input=scaled, pool_type="sum")
        return weighted_memory, weights
