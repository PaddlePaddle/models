#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
lstm class
"""
import paddle_layers as layers
from paddle.fluid.dygraph import Layer, Linear
from paddle import fluid
from utils import seq_length


class LSTM(Layer):
    """
    LSTM
    """

    def __init__(self, conf_dict):
        """
        initialize
        """
        super(LSTM, self).__init__()
        self.dict_size = conf_dict["dict_size"]
        self.task_mode = conf_dict["task_mode"]
        self.emb_dim = conf_dict["net"]["emb_dim"]
        self.lstm_dim = conf_dict["net"]["lstm_dim"]
        self.hidden_dim = conf_dict["net"]["hidden_dim"]
        self.emb_layer = layers.EmbeddingLayer(self.dict_size, self.emb_dim,
                                               "emb").ops()
        self.lstm_layer = layers.DynamicLSTMLayer(self.lstm_dim, "lstm").ops()
        self.fc_layer = layers.FCLayer(self.hidden_dim, None, "fc").ops()
        self.softmax_layer = layers.FCLayer(2, "softmax", "cos_sim").ops()
        self.proj_layer = Linear(
            input_dim=self.hidden_dim, output_dim=self.lstm_dim * 4)
        self.last_layer = layers.ExtractLastLayer()
        self.seq_len = conf_dict["seq_len"]

    def forward(self, left, right):
        """
        Forward network
        """
        # embedding layer
        left_emb = self.emb_layer(left)
        right_emb = self.emb_layer(right)
        # Presentation context
        left_proj = self.proj_layer(left_emb)
        right_proj = self.proj_layer(right_emb)
        left_lstm, _ = self.lstm_layer(left_proj)
        right_lstm, _ = self.lstm_layer(right_proj)
        # Get sequence length before padding
        left_len = seq_length(left)
        left_len.stop_gradient = True
        right_len = seq_length(right)
        right_len.stop_gradient = True
        # Extract last step
        left_last = self.last_layer.ops(left_lstm, left_len)
        right_last = self.last_layer.ops(right_lstm, right_len)

        # matching layer
        if self.task_mode == "pairwise":
            left_fc = self.fc_layer(left_last)
            right_fc = self.fc_layer(right_last)
            cos_sim_layer = layers.CosSimLayer()
            pred = cos_sim_layer.ops(left_fc, right_fc)
            return left_fc, pred
        else:
            concat_layer = layers.ConcatLayer(1)
            concat = concat_layer.ops([left_last, right_last])
            concat_fc = self.fc_layer(concat)
            pred = self.softmax_layer(concat_fc)
            return left_last, pred
