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
MMDNN class
"""
import numpy as np
import paddle.fluid as fluid
import logging
from paddle.fluid.dygraph import Linear, to_variable, Layer, Pool2D, Conv2D
import paddle_layers as pd_layers
from paddle.fluid import layers


class MMDNN(Layer):
    """
    MMDNN
    """

    def __init__(self, config):
        """
        initialize
        """
        super(MMDNN, self).__init__()

        self.vocab_size = int(config['dict_size'])
        self.emb_size = int(config['net']['embedding_dim'])
        self.lstm_dim = int(config['net']['lstm_dim'])
        self.kernel_size = int(config['net']['num_filters'])
        self.win_size1 = int(config['net']['window_size_left'])
        self.win_size2 = int(config['net']['window_size_right'])
        self.dpool_size1 = int(config['net']['dpool_size_left'])
        self.dpool_size2 = int(config['net']['dpool_size_right'])
        self.hidden_size = int(config['net']['hidden_size'])
        self.seq_len = int(config["seq_len"])
        self.seq_len1 = self.seq_len
        #int(config['max_len_left'])
        self.seq_len2 = self.seq_len 
        #int(config['max_len_right'])
        self.task_mode = config['task_mode']
        self.zero_pad = True
        self.scale = False

        if int(config['match_mask']) != 0:
            self.match_mask = True
        else:
            self.match_mask = False

        if self.task_mode == "pointwise":
            self.n_class = int(config['n_class'])
            self.out_size = self.n_class
        elif self.task_mode == "pairwise":
            self.out_size = 1
        else:
            logging.error("training mode not supported")

        # layers
        self.emb_layer = pd_layers.EmbeddingLayer(self.vocab_size, self.emb_size, 
            name="word_embedding",padding_idx=(0 if self.zero_pad else None)).ops()
        self.fw_in_proj = Linear(
            input_dim=self.emb_size,
            output_dim=4 * self.lstm_dim,
            param_attr=fluid.ParamAttr(name="fw_fc.w"),
            bias_attr=False)
        self.lstm_layer = pd_layers.DynamicLSTMLayer(self.lstm_dim, "lstm").ops()
        self.rv_in_proj = Linear(
            input_dim=self.emb_size,
            output_dim=4 * self.lstm_dim,
            param_attr=fluid.ParamAttr(name="rv_fc.w"),
            bias_attr=False)
        self.reverse_layer = pd_layers.DynamicLSTMLayer(
            self.lstm_dim,
            is_reverse=True).ops()
  
        self.conv = Conv2D(
            num_channels=1,
            num_filters=self.kernel_size,
            stride=1,
            padding=(int(self.seq_len1 / 2), int(self.seq_len2 // 2)),
            filter_size=(self.seq_len1, self.seq_len2),
            bias_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Constant(0.1)))
     
        self.pool_layer = Pool2D(
            pool_size=[
                int(self.seq_len1 / self.dpool_size1),
                int(self.seq_len2 / self.dpool_size2)
            ],
            pool_stride=[
                int(self.seq_len1 / self.dpool_size1),
                int(self.seq_len2 / self.dpool_size2)
            ],
            pool_type="max" )
        self.fc_layer = pd_layers.FCLayer(self.hidden_size, "tanh", "fc").ops()
        self.fc1_layer = pd_layers.FCLayer(self.out_size, "softmax", "fc1").ops()
        


    def forward(self, left, right):
        """
        Forward network
        """
        left_emb = self.emb_layer(left)
        right_emb = self.emb_layer(right)
        if self.scale:
            left_emb = left_emb * (self.emb_size**0.5)
            right_emb = right_emb * (self.emb_size**0.5)

        # bi_listm
        left_proj = self.fw_in_proj(left_emb)
        right_proj = self.fw_in_proj(right_emb)

        left_lstm, _ = self.lstm_layer(left_proj)
        right_lstm, _ = self.lstm_layer(right_proj)
        left_rv_proj = self.rv_in_proj(left_lstm)
        right_rv_proj = self.rv_in_proj(right_lstm)
        left_reverse,_ = self.reverse_layer(left_rv_proj)
        right_reverse,_ = self.reverse_layer(right_rv_proj)
   
        left_seq_encoder = fluid.layers.concat([left_lstm, left_reverse], axis=1)
        right_seq_encoder = fluid.layers.concat([right_lstm, right_reverse], axis=1)
  
        pad_value = fluid.layers.assign(input=np.array([0]).astype("float32"))
        left_seq_encoder = fluid.layers.reshape(left_seq_encoder, shape=[int(left_seq_encoder.shape[0]/self.seq_len),self.seq_len,-1])
        right_seq_encoder = fluid.layers.reshape(right_seq_encoder, shape=[int(right_seq_encoder.shape[0]/self.seq_len),self.seq_len,-1])
        cross = fluid.layers.matmul(
            left_seq_encoder, right_seq_encoder, transpose_y=True)
  
        left_lens=to_variable(np.array([self.seq_len]))
        right_lens=to_variable(np.array([self.seq_len]))


        if self.match_mask:
            mask1 = fluid.layers.sequence_mask(
                x=left_lens, dtype='float32', maxlen=self.seq_len1 + 1)
            mask2 = fluid.layers.sequence_mask(
                x=right_lens, dtype='float32', maxlen=self.seq_len2 + 1)
         
            mask1 = fluid.layers.transpose(x=mask1, perm=[1, 0])
            mask = fluid.layers.matmul(x=mask1, y=mask2)
        else:
            mask = None

        # conv_pool_relu
        emb_expand = fluid.layers.unsqueeze(input=cross, axes=[1])

        conv = self.conv(emb_expand)
        if mask is not None:
            cross_mask = fluid.layers.stack(x=[mask] * self.kernel_size, axis=0)
            cross_mask = fluid.layers.stack(x=[cross_mask] * conv.shape[0], axis=0)
            conv = cross_mask * conv + (1 - cross_mask) * (-2**self.seq_len + 1)

        pool = self.pool_layer(conv)
        conv_pool_relu = fluid.layers.relu(pool)

        relu_hid1 = self.fc_layer(conv_pool_relu)
        relu_hid1 = fluid.layers.tanh(relu_hid1)

        pred = self.fc1_layer(relu_hid1)
        pred = fluid.layers.softmax(pred)
        return left_seq_encoder, pred
