# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved
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
import paddle.fluid.layers.nn as nn
import paddle.fluid.layers.tensor as tensor
import paddle.fluid.layers.control_flow as cf
import paddle.fluid.layers.io as io


class BowEncoder(object):
    """ bow-encoder """

    def __init__(self):
        self.param_name = ""

    def forward(self, emb):
        return nn.sequence_pool(input=emb, pool_type='sum')


class CNNEncoder(object):
    """ cnn-encoder"""

    def __init__(self,
                 param_name="cnn",
                 win_size=3,
                 ksize=128,
                 act='tanh',
                 pool_type='max'):
        self.param_name = param_name
        self.win_size = win_size
        self.ksize = ksize
        self.act = act
        self.pool_type = pool_type

    def forward(self, emb):
        return fluid.nets.sequence_conv_pool(
            input=emb,
            num_filters=self.ksize,
            filter_size=self.win_size,
            act=self.act,
            pool_type=self.pool_type,
            param_attr=self.param_name + ".param",
            bias_attr=self.param_name + ".bias")
        


class GrnnEncoder(object):
    """ grnn-encoder """

    def __init__(self, param_name="grnn", hidden_size=128):
        self.param_name = param_name
        self.hidden_size = hidden_size

    def forward(self, emb):
        fc0 = nn.fc(
            input=emb, 
            size=self.hidden_size * 3, 
            param_attr=self.param_name + "_fc.w",
            bias_attr=False)
        
        gru_h = nn.dynamic_gru(
            input=fc0,
            size=self.hidden_size,
            is_reverse=False,
            param_attr=self.param_name + ".param",
            bias_attr=self.param_name + ".bias")
        return nn.sequence_pool(input=gru_h, pool_type='max')


'''this is a very simple Encoder factory
most default argument values are used'''


class SimpleEncoderFactory(object):
    def __init__(self):
        pass

    ''' create an encoder through create function '''

    def create(self, enc_type, enc_hid_size):
        if enc_type == "bow":
            bow_encode = BowEncoder()
            return bow_encode
        elif enc_type == "cnn":
            cnn_encode = CNNEncoder(ksize=enc_hid_size)
            return cnn_encode
        elif enc_type == "gru":
            rnn_encode = GrnnEncoder(hidden_size=enc_hid_size)
            return rnn_encode


class MultiviewSimnet(object):
    """ multi-view simnet """

    def __init__(self, embedding_size, embedding_dim, hidden_size):
        self.embedding_size = embedding_size
        self.embedding_dim = embedding_dim
        self.emb_shape = [self.embedding_size, self.embedding_dim]
        self.hidden_size = hidden_size
        self.margin = 0.1

    def set_query_encoder(self, encoders):
        self.query_encoders = encoders

    def set_title_encoder(self, encoders):
        self.title_encoders = encoders

    def get_correct(self, x, y):
        less = tensor.cast(cf.less_than(x, y), dtype='float32')
        correct = nn.reduce_sum(less)
        return correct

    def train_net(self):
        # input fields for query, pos_title, neg_title
        q_slots = [
            io.data(
                name="q%d" % i, shape=[1], lod_level=1, dtype='int64')
            for i in range(len(self.query_encoders))
        ]
        pt_slots = [
            io.data(
                name="pt%d" % i, shape=[1], lod_level=1, dtype='int64')
            for i in range(len(self.title_encoders))
        ]
        nt_slots = [
            io.data(
                name="nt%d" % i, shape=[1], lod_level=1, dtype='int64')
            for i in range(len(self.title_encoders))
        ]

        # lookup embedding for each slot
        q_embs = [
            nn.embedding(
                input=query, size=self.emb_shape, param_attr="emb")
            for query in q_slots
        ]
        pt_embs = [
            nn.embedding(
                input=title, size=self.emb_shape, param_attr="emb")
            for title in pt_slots
        ]
        nt_embs = [
            nn.embedding(
                input=title, size=self.emb_shape, param_attr="emb")
            for title in nt_slots
        ]

        # encode each embedding field with encoder
        q_encodes = [
            self.query_encoders[i].forward(emb) for i, emb in enumerate(q_embs)
        ]
        pt_encodes = [
            self.title_encoders[i].forward(emb) for i, emb in enumerate(pt_embs)
        ]
        nt_encodes = [
            self.title_encoders[i].forward(emb) for i, emb in enumerate(nt_embs)
        ]

        # concat multi view for query, pos_title, neg_title
        q_concat = nn.concat(q_encodes)
        pt_concat = nn.concat(pt_encodes)
        nt_concat = nn.concat(nt_encodes)

        # projection of hidden layer
        q_hid = nn.fc(q_concat, size=self.hidden_size, param_attr='q_fc.w', bias_attr='q_fc.b')
        pt_hid = nn.fc(pt_concat, size=self.hidden_size, param_attr='t_fc.w', bias_attr='t_fc.b')
        nt_hid = nn.fc(nt_concat, size=self.hidden_size, param_attr='t_fc.w', bias_attr='t_fc.b')

        # cosine of hidden layers
        cos_pos = nn.cos_sim(q_hid, pt_hid)
        cos_neg = nn.cos_sim(q_hid, nt_hid)

        # pairwise hinge_loss
        loss_part1 = nn.elementwise_sub(
            tensor.fill_constant_batch_size_like(
                input=cos_pos,
                shape=[-1, 1],
                value=self.margin,
                dtype='float32'),
            cos_pos)

        loss_part2 = nn.elementwise_add(loss_part1, cos_neg)

        loss_part3 = nn.elementwise_max(
            tensor.fill_constant_batch_size_like(
                input=loss_part2, shape=[-1, 1], value=0.0, dtype='float32'),
            loss_part2)

        avg_cost = nn.mean(loss_part3)
        correct = self.get_correct(cos_neg, cos_pos)

        return q_slots + pt_slots + nt_slots, avg_cost, correct

    def pred_net(self, query_fields, pos_title_fields, neg_title_fields):
        q_slots = [
            io.data(
                name="q%d" % i, shape=[1], lod_level=1, dtype='int64')
            for i in range(len(self.query_encoders))
        ]
        pt_slots = [
            io.data(
                name="pt%d" % i, shape=[1], lod_level=1, dtype='int64')
            for i in range(len(self.title_encoders))
        ]
        # lookup embedding for each slot
        q_embs = [
            nn.embedding(
                input=query, size=self.emb_shape, param_attr="emb")
            for query in q_slots
        ]
        pt_embs = [
            nn.embedding(
                input=title, size=self.emb_shape, param_attr="emb")
            for title in pt_slots
        ]
        # encode each embedding field with encoder
        q_encodes = [
            self.query_encoder[i].forward(emb) for i, emb in enumerate(q_embs)
        ]
        pt_encodes = [
            self.title_encoders[i].forward(emb) for i, emb in enumerate(pt_embs)
        ]
        # concat multi view for query, pos_title, neg_title
        q_concat = nn.concat(q_encodes)
        pt_concat = nn.concat(pt_encodes)
        # projection of hidden layer
        q_hid = nn.fc(q_concat, size=self.hidden_size, param_attr='q_fc.w', bias_attr='q_fc.b')
        pt_hid = nn.fc(pt_concat, size=self.hidden_size, param_attr='t_fc.w', bias_attr='t_fc.b')
        # cosine of hidden layers
        cos = nn.cos_sim(q_hid, pt_hid)
        return cos
