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
cnn class
"""

import paddle_layers as layers
from paddle import fluid
from paddle.fluid.dygraph import Layer

class CNN(Layer):
    """
    CNN
    """

    def __init__(self, conf_dict):
        """
        initialize
        """
        super(CNN, self).__init__()
        self.dict_size = conf_dict["dict_size"]
        self.task_mode = conf_dict["task_mode"]
        self.emb_dim = conf_dict["net"]["emb_dim"]
        self.filter_size = conf_dict["net"]["filter_size"]
        self.num_filters = conf_dict["net"]["num_filters"]
        self.hidden_dim = conf_dict["net"]["hidden_dim"]
        self.seq_len = conf_dict["seq_len"]
        self.channels = 1
        
        # layers
        self.emb_layer = layers.EmbeddingLayer(self.dict_size, self.emb_dim, "emb").ops()
        self.fc_layer = layers.FCLayer(self.hidden_dim, None, "fc").ops()
        self.softmax_layer = layers.FCLayer(2, "softmax", "cos_sim").ops()
        self.cnn_layer = layers.SimpleConvPool(
            self.channels,
            self.num_filters,
            self.filter_size)
    

    def forward(self, left, right):
        """
        Forward network
        """
        # embedding layer 
       
        left_emb = self.emb_layer(left)
        right_emb = self.emb_layer(right)
        # Presentation context

        left_emb = fluid.layers.reshape(
            left_emb, shape=[-1, self.channels, self.seq_len, self.hidden_dim])
        right_emb = fluid.layers.reshape(
            right_emb, shape=[-1, self.channels, self.seq_len, self.hidden_dim])
    
        left_cnn = self.cnn_layer(left_emb)
        right_cnn = self.cnn_layer(right_emb)
        # matching layer
        if self.task_mode == "pairwise":
            left_fc = self.fc_layer(left_cnn)
            right_fc = self.fc_layer(right_cnn)
            cos_sim_layer = layers.CosSimLayer()
            pred = cos_sim_layer.ops(left_fc, right_fc)
            return left_fc, pred
        else:
            concat_layer = layers.ConcatLayer(1)
            concat = concat_layer.ops([left_cnn, right_cnn])
            concat_fc = self.fc_layer(concat)
            pred = self.softmax_layer(concat_fc)
            return left_cnn, pred
