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
from .my_layers import bi_lstm_layer
from .match_layers import ElementwiseMatching


class SSENet():
    """
    SSE net: Shortcut-Stacked Sentence Encoders for Multi-Domain Inference
    https://arxiv.org/abs/1708.02312
    """

    def __init__(self, config):
        self._config = config

    def __call__(self, seq1, seq2, label):
        return self.body(seq1, seq2, label, self._config)

    def body(self, seq1, seq2, label, config):
        """Body function"""

        def stacked_bi_rnn_model(seq):
            embed = fluid.layers.embedding(
                input=seq,
                size=[self._config.dict_dim, self._config.emb_dim],
                param_attr='emb.w')
            stacked_lstm_out = [embed]
            for i in range(len(self._config.rnn_hid_dim)):
                if i == 0:
                    feature = embed
                else:
                    feature = fluid.layers.concat(
                        input=stacked_lstm_out, axis=1)
                bi_lstm_h = bi_lstm_layer(
                    feature,
                    rnn_hid_dim=self._config.rnn_hid_dim[i],
                    name="lstm_" + str(i))

                # add dropout except for the last stacked lstm layer
                if i != len(self._config.rnn_hid_dim) - 1:
                    bi_lstm_h = fluid.layers.dropout(
                        bi_lstm_h, dropout_prob=self._config.droprate_lstm)
                stacked_lstm_out.append(bi_lstm_h)
            pool = fluid.layers.sequence_pool(input=bi_lstm_h, pool_type='max')
            return pool

        def MLP(vec):
            for i in range(len(self._config.fc_dim)):
                vec = fluid.layers.fc(vec,
                                      size=self._config.fc_dim[i],
                                      act='relu')
                # add dropout after every layer of MLP
                vec = fluid.layers.dropout(
                    vec, dropout_prob=self._config.droprate_fc)
            return vec

        seq1_rnn = stacked_bi_rnn_model(seq1)
        seq2_rnn = stacked_bi_rnn_model(seq2)
        seq_match = ElementwiseMatching(seq1_rnn, seq2_rnn)

        mlp_res = MLP(seq_match)
        prediction = fluid.layers.fc(mlp_res,
                                     size=self._config.class_dim,
                                     act='softmax')
        loss = fluid.layers.cross_entropy(input=prediction, label=label)
        avg_cost = fluid.layers.mean(x=loss)
        acc = fluid.layers.accuracy(input=prediction, label=label)
        return avg_cost, acc, prediction
