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


class GRUEncoder(object):
    def __init__(self,
                 hidden_dim,
                 num_layers=1,
                 bidirectional=True,
                 dropout=0.0,
                 name=None):
        num_directions = 2 if bidirectional else 1
        assert hidden_dim % num_directions == 0
        rnn_hidden_dim = hidden_dim // num_directions

        self.hidden_dim = hidden_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout

        self.gru = layers.GRU(hidden_dim=rnn_hidden_dim,
                              num_layers=num_layers,
                              bidirectional=bidirectional,
                              dropout=dropout if self.num_layers > 1 else 0.0,
                              name=name)

    def __call__(self, inputs, hidden=None):
        outputs, new_hidden = self.gru(inputs, hidden)
        return outputs, new_hidden
