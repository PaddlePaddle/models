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

import paddle.fluid as fluid
import my_layers
import numpy

class ESIMNet():
    """
    from paper: Enhanced Sequential for Natural Language Inference Model
    """

    def __init__(self, config):
         self._config = config

    def __call__(self, seq1, seq2, label):
        """
        seq1: lod tensor in shape [-1, 1], lod_level = 1
        seq2: lod tensor in shape [-1, 1], lod_level = 1
        label: tensor in shape [batch_size, 1]
        """
        return self.body(seq1, seq2, label)

    def body(self, seq1, seq2, label):
        """Body function"""

        # embdding
        embed1 = self.embedding(seq1)
        embed1 = fluid.layers.dropout(embed1, dropout_prob=self._config.drop_rate)
        embed2 = self.embedding(seq2)
        embed2 = fluid.layers.dropout(embed2, dropout_prob=self._config.drop_rate)

        # first bi-lstm
        first_rnn1 = fluid.layers.dropout(
                         self.bi_lstm(embed1, name="first_lstm"), 
                         dropout_prob=self._config.drop_rate)
        first_rnn2 = fluid.layers.dropout(
                         self.bi_lstm(embed2, name="first_lstm"), 
                         dropout_prob=self._config.drop_rate)
        
        # lod-tensor to tensor
        pad_value = fluid.layers.assign(input=numpy.array([0], dtype=numpy.float32))
        padded_rnn1, mask1 = fluid.layers.sequence_pad(x=first_rnn1, pad_value=pad_value)
        padded_rnn2, mask2 = fluid.layers.sequence_pad(x=first_rnn2, pad_value=pad_value)

        # soft-aligned attention
        att1, att2 = self.attention(padded_rnn1, padded_rnn2)
        
        # match layer
        m1 = self.concat_match(padded_rnn1, att1)
        m2 = self.concat_match(padded_rnn2, att2)

        # tensor 2 lod-tensor
        m1 = fluid.layers.sequence_unpad(x=m1, length=mask1)
        m2 = fluid.layers.sequence_unpad(x=m2, length=mask2)

        # second bi-lstm
        second_rnn1 = fluid.layers.dropout(
                          self.bi_lstm(m1, name="second_lstm"), 
                          dropout_prob=self._config.drop_rate)
        second_rnn2 = fluid.layers.dropout(
                          self.bi_lstm(m2, name="second_lstm"), 
                          dropout_prob=self._config.drop_rate)

        # pooling
        max_pool1 = fluid.layers.sequence_pool(second_rnn1, pool_type='max')
        avg_pool1 = fluid.layers.sequence_pool(second_rnn1, pool_type='average')
        max_pool2 = fluid.layers.sequence_pool(second_rnn2, pool_type='max')
        avg_pool2 = fluid.layers.sequence_pool(second_rnn2, pool_type='average')
        pool = fluid.layers.concat([avg_pool1, max_pool1, avg_pool2, max_pool2], axis=1)

        # mlp
        fc = fluid.layers.fc(pool, size=self._config.mlp_hid_dim, act="tanh")
        fc = fluid.layers.dropout(fc, dropout_prob=self._config.drop_rate)

        # loss layer
        prediction = fluid.layers.fc(fc, size=self._config.class_dim, act='softmax')
        loss = fluid.layers.cross_entropy(input=prediction, label=label)
        avg_cost = fluid.layers.mean(x=loss)
        acc = fluid.layers.accuracy(input=prediction, label=label)
        return avg_cost, acc, prediction

    def concat_match(self, vec1, vec2):
        """
        concat match for two vectors
        vec1 and vec2 is in shape: [batch_size, seq_len, hid_dim]
        return vec in shape [batch_size, seq_len, hid_dim * 4]
        """
        return fluid.layers.concat(
                   input=[vec1,
                          vec2,
                          fluid.layers.elementwise_sub(x=vec1, y=vec2),
                          fluid.layers.elementwise_mul(x=vec1, y=vec2)],
                   axis=2)

    def attention(self, seq1, seq2):
        """
        seq1 and seq2 are two sequence with padding
        return beta: seq1's attention on seq2
        return alpha: seq2's attention on seq1
        """
        attention_weight = fluid.layers.matmul(seq1, seq2, transpose_y=True)
        normalized_attention_weight = fluid.layers.softmax(attention_weight)
        beta = fluid.layers.matmul(normalized_attention_weight, seq2)
        attention_weight_t = fluid.layers.transpose(attention_weight, perm=[0, 2, 1])
        normalized_attention_weight_t = fluid.layers.softmax(attention_weight_t)
        alpha = fluid.layers.matmul(normalized_attention_weight_t, seq1)
        return beta, alpha

    def embedding(self, x):
        """
        x: lod tensor in shape [-1, 1], lod_level = 1
        return embed: lod tensor in shape [-1, emb_dim], lod_level = 1
        """
        return fluid.layers.embedding(
                   input=x,
                   size=[self._config.dict_dim, self._config.emb_dim],
                   param_attr=fluid.ParamAttr(name='emb.w'))

    def bi_lstm(self, x, name):
        """
        x: lod tensor in shape [-1, emb_dim], lod_level = 1
        """
        return my_layers.bi_lstm_layer(
                   input=x,
                   rnn_hid_dim=self._config.lstm_hid_dim,
                   name=name)

