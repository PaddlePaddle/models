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
bow class
"""

import paddle_layers as layers
from paddle import fluid
from paddle.fluid.dygraph.base import to_variable
from paddle.fluid.dygraph import Layer, Linear
import paddle.fluid.param_attr as attr

class BOW(Layer):
    """
    BOW
    """

    def __init__(self, conf_dict):
        """
        initialize
        """
        super(BOW, self).__init__()
        self.dict_size = conf_dict["dict_size"]
        self.task_mode = conf_dict["task_mode"]
        self.emb_dim = conf_dict["net"]["emb_dim"]
        self.bow_dim = conf_dict["net"]["bow_dim"]
        self.seq_len = conf_dict["seq_len"]
        self.emb_layer = layers.EmbeddingLayer(self.dict_size, self.emb_dim, "emb").ops()
        self.bow_layer = Linear(self.bow_dim, self.bow_dim)
        self.bow_layer_po = layers.FCLayer(self.bow_dim, None, "fc").ops()
        self.softmax_layer = layers.FCLayer(2, "softmax", "cos_sim").ops()
    

    def forward(self, left, right):
        """
        Forward network
        """
        
        # embedding layer
        left_emb = self.emb_layer(left)
        right_emb = self.emb_layer(right)
        left_emb = fluid.layers.reshape(
            left_emb, shape=[-1, self.seq_len, self.bow_dim])
        right_emb = fluid.layers.reshape(
            right_emb, shape=[-1, self.seq_len, self.bow_dim])
       
        bow_left = fluid.layers.reduce_sum(left_emb, dim=1)
        bow_right = fluid.layers.reduce_sum(right_emb, dim=1)
        softsign_layer = layers.SoftsignLayer()
        left_soft = softsign_layer.ops(bow_left)
        right_soft = softsign_layer.ops(bow_right)
      
        # matching layer
        if self.task_mode == "pairwise":
            left_bow = self.bow_layer(left_soft)
            right_bow = self.bow_layer(right_soft)
            cos_sim_layer = layers.CosSimLayer()
            pred = cos_sim_layer.ops(left_bow, right_bow)
            return left_bow, pred
        else:
            concat_layer = layers.ConcatLayer(1)
            concat = concat_layer.ops([left_soft, right_soft])
            concat_fc = self.bow_layer_po(concat)
            pred = self.softmax_layer(concat_fc)
            return left_soft, pred
