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


class DecAttNet():
    """decompose attention net"""

    def __init__(self, config):
        self._config = config
        self.initializer = fluid.initializer.Xavier(uniform=False)

    def __call__(self, seq1, seq2, mask1, mask2, label):
        return self.body(seq1, seq2, mask1, mask2, label)

    def body(self, seq1, seq2, mask1, mask2, label):
        """Body function"""
        transformed_q1 = self.transformation(seq1)
        transformed_q2 = self.transformation(seq2)
        masked_q1 = self.apply_mask(transformed_q1, mask1)
        masked_q2 = self.apply_mask(transformed_q2, mask2)
        alpha, beta = self.attend(masked_q1, masked_q2)
        if self._config.share_wight_btw_seq:
            seq1_compare = self.compare(masked_q1, beta, param_prefix='compare')
            seq2_compare = self.compare(
                masked_q2, alpha, param_prefix='compare')
        else:
            seq1_compare = self.compare(
                masked_q1, beta, param_prefix='compare_1')
            seq2_compare = self.compare(
                masked_q2, alpha, param_prefix='compare_2')
        aggregate_res = self.aggregate(seq1_compare, seq2_compare)
        prediction = fluid.layers.fc(aggregate_res,
                                     size=self._config.class_dim,
                                     act='softmax')
        loss = fluid.layers.cross_entropy(input=prediction, label=label)
        avg_cost = fluid.layers.mean(x=loss)
        acc = fluid.layers.accuracy(input=prediction, label=label)
        return avg_cost, acc, prediction

    def apply_mask(self, seq, mask):
        """
       apply mask on seq
       Input: seq in shape [batch_size, seq_len, embedding_size]
       Input: mask in shape [batch_size, seq_len]
       Output: masked seq in shape [batch_size, seq_len, embedding_size]
       """
        return fluid.layers.elementwise_mul(x=seq, y=mask, axis=0)

    def feed_forward_2d(self, vec, param_prefix):
        """
        Input: vec in shape [batch_size, seq_len, vec_dim]
        Output: fc2 in shape [batch_size, seq_len, num_units[1]]
        """
        fc1 = fluid.layers.fc(vec,
                              size=self._config.num_units[0],
                              num_flatten_dims=2,
                              param_attr=fluid.ParamAttr(
                                  name=param_prefix + '_fc1.w',
                                  initializer=self.initializer),
                              bias_attr=param_prefix + '_fc1.b',
                              act='relu')
        fc1 = fluid.layers.dropout(fc1, dropout_prob=self._config.droprate)
        fc2 = fluid.layers.fc(fc1,
                              size=self._config.num_units[1],
                              num_flatten_dims=2,
                              param_attr=fluid.ParamAttr(
                                  name=param_prefix + '_fc2.w',
                                  initializer=self.initializer),
                              bias_attr=param_prefix + '_fc2.b',
                              act='relu')
        fc2 = fluid.layers.dropout(fc2, dropout_prob=self._config.droprate)
        return fc2

    def feed_forward(self, vec, param_prefix):
        """
        Input: vec in shape [batch_size, vec_dim]
        Output: fc2 in shape [batch_size, num_units[1]]
        """
        fc1 = fluid.layers.fc(vec,
                              size=self._config.num_units[0],
                              num_flatten_dims=1,
                              param_attr=fluid.ParamAttr(
                                  name=param_prefix + '_fc1.w',
                                  initializer=self.initializer),
                              bias_attr=param_prefix + '_fc1.b',
                              act='relu')
        fc1 = fluid.layers.dropout(fc1, dropout_prob=self._config.droprate)
        fc2 = fluid.layers.fc(fc1,
                              size=self._config.num_units[1],
                              num_flatten_dims=1,
                              param_attr=fluid.ParamAttr(
                                  name=param_prefix + '_fc2.w',
                                  initializer=self.initializer),
                              bias_attr=param_prefix + '_fc2.b',
                              act='relu')
        fc2 = fluid.layers.dropout(fc2, dropout_prob=self._config.droprate)
        return fc2

    def transformation(self, seq):
        embed = fluid.layers.embedding(
            input=seq,
            size=[self._config.dict_dim, self._config.emb_dim],
            param_attr=fluid.ParamAttr(
                name='emb.w', trainable=self._config.word_embedding_trainable))
        if self._config.proj_emb_dim is not None:
            return fluid.layers.fc(embed,
                                   size=self._config.proj_emb_dim,
                                   num_flatten_dims=2,
                                   param_attr=fluid.ParamAttr(
                                       name='project' + '_fc1.w',
                                       initializer=self.initializer),
                                   bias_attr=False,
                                   act=None)
        return embed

    def attend(self, seq1, seq2):
        """
        Input: seq1, shape [batch_size, seq_len1, embed_size]
        Input: seq2, shape [batch_size, seq_len2, embed_size]
        Output: alpha, shape [batch_size, seq_len1, embed_size]
        Output: beta, shape [batch_size, seq_len2, embed_size]
        """
        if self._config.share_wight_btw_seq:
            seq1 = self.feed_forward_2d(seq1, param_prefix="attend")
            seq2 = self.feed_forward_2d(seq2, param_prefix="attend")
        else:
            seq1 = self.feed_forward_2d(seq1, param_prefix="attend_1")
            seq2 = self.feed_forward_2d(seq2, param_prefix="attend_2")
        attention_weight = fluid.layers.matmul(seq1, seq2, transpose_y=True)
        normalized_attention_weight = fluid.layers.softmax(attention_weight)
        beta = fluid.layers.matmul(normalized_attention_weight, seq2)
        attention_weight_t = fluid.layers.transpose(
            attention_weight, perm=[0, 2, 1])
        normalized_attention_weight_t = fluid.layers.softmax(attention_weight_t)
        alpha = fluid.layers.matmul(normalized_attention_weight_t, seq1)
        return alpha, beta

    def compare(self, seq, soft_alignment, param_prefix):
        concat_seq = fluid.layers.concat(input=[seq, soft_alignment], axis=2)
        return self.feed_forward_2d(concat_seq, param_prefix="compare")

    def aggregate(self, vec1, vec2):
        vec1 = fluid.layers.reduce_sum(vec1, dim=1)
        vec2 = fluid.layers.reduce_sum(vec2, dim=1)
        concat_vec = fluid.layers.concat(input=[vec1, vec2], axis=1)
        return self.feed_forward(concat_vec, param_prefix='aggregate')
