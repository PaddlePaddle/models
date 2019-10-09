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
from mmpms.modules.attention import Attention


class BaseDecoder(object):
    def step(self, input, state):
        """ step function """
        raise NotImplementedError

    def forward(self, input, state):
        """ forward function """
        drnn = layers.DynamicRNN()

        def memory(memory_state):
            if isinstance(memory_state, dict):
                return {k: memory(v) for k, v in memory_state.items()}
            elif isinstance(memory_state, (tuple, list)):
                return type(memory_state)(memory(x) for x in memory_state)
            else:
                return drnn.memory(init=memory_state, need_reorder=True)

        def update(pre_state, new_state):
            if isinstance(new_state, dict):
                for k in new_state.keys():
                    if k in pre_state:
                        update(pre_state[k], new_state[k])
            elif isinstance(new_state, (tuple, list)):
                for i in range(len(new_state)):
                    update(pre_state[i], new_state[i])
            else:
                drnn.update_memory(pre_state, new_state)

        with drnn.block():
            current_input = drnn.step_input(input)
            pre_state = memory(state)
            output, current_state = self.step(current_input, pre_state)
            update(pre_state, current_state)
            drnn.output(output)
        rnn_output = drnn()
        return rnn_output

    def __call__(self, input, state):
        return self.forward(input, state)


class GRUDecoder(BaseDecoder):
    def __init__(self,
                 hidden_dim,
                 num_layers=1,
                 attn_mode="none",
                 attn_hidden_dim=None,
                 memory_dim=None,
                 dropout=0.0,
                 name=None):
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.attn_mode = None if attn_mode == "none" else attn_mode
        self.attn_hidden_dim = attn_hidden_dim or hidden_dim // 2
        self.memory_dim = memory_dim or hidden_dim
        self.dropout = dropout

        self.rnn = layers.StackedGRUCell(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if self.num_layers > 1 else 0.0,
            name=name)

        if self.attn_mode:
            self.attention = Attention(
                mode=self.attn_mode,
                memory_dim=self.memory_dim,
                hidden_dim=self.attn_hidden_dim)

    def step(self, input, state):
        hidden = state["hidden"]
        rnn_input_list = [input]

        if self.attn_mode:
            memory = state["memory"]
            memory_proj = state.get("memory_proj")
            query = hidden[-1]
            context, _ = self.attention(
                query=query, memory=memory, memory_proj=memory_proj)
            rnn_input_list.append(context)

        rnn_input = layers.concat(rnn_input_list, axis=1)
        rnn_output, new_hidden = self.rnn(rnn_input, hidden)

        new_state = {k: v for k, v in state.items() if k != "hidden"}
        new_state["hidden"] = new_hidden
        if self.attn_mode:
            output = layers.concat([rnn_output, context], axis=1)
        else:
            output = rnn_output
        return output, new_state
