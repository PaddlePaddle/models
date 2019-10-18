# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved
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
"""
This module defines some Frequently-used DNN layers
"""

import paddle.fluid as fluid


def bi_lstm_layer(input, rnn_hid_dim, name):
    """
    This is a Bi-directional LSTM(long short term memory) Module
    """
    fc0 = fluid.layers.fc(
        input=input,  # fc for lstm
        size=rnn_hid_dim * 4,
        param_attr=name + '.fc0.w',
        bias_attr=False,
        act=None)

    lstm_h, c = fluid.layers.dynamic_lstm(
        input=fc0,
        size=rnn_hid_dim * 4,
        is_reverse=False,
        param_attr=name + '.lstm_w',
        bias_attr=name + '.lstm_b')

    reversed_lstm_h, reversed_c = fluid.layers.dynamic_lstm(
        input=fc0,
        size=rnn_hid_dim * 4,
        is_reverse=True,
        param_attr=name + '.reversed_lstm_w',
        bias_attr=name + '.reversed_lstm_b')
    return fluid.layers.concat(input=[lstm_h, reversed_lstm_h], axis=1)
