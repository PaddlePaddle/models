#Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved
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


class GrnnEncoder(object):
    """ grnn-encoder """

    def __init__(self, param_name="grnn", hidden_size=128):
        self.param_name = param_name
        self.hidden_size = hidden_size

    def forward(self, emb):
        fc0 = nn.fc(input=emb,
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


class PairwiseHingeLoss(object):
    def __init__(self, margin=0.8):
        self.margin = margin

    def forward(self, pos, neg):
        loss_part1 = nn.elementwise_sub(
            tensor.fill_constant_batch_size_like(
                input=pos, shape=[-1, 1], value=self.margin, dtype='float32'),
            pos)
        loss_part2 = nn.elementwise_add(loss_part1, neg)
        loss_part3 = nn.elementwise_max(
            tensor.fill_constant_batch_size_like(
                input=loss_part2, shape=[-1, 1], value=0.0, dtype='float32'),
            loss_part2)
        return loss_part3


class SequenceSemanticRetrieval(object):
    """ sequence semantic retrieval model """

    def __init__(self, embedding_size, embedding_dim, hidden_size):
        self.embedding_size = embedding_size
        self.embedding_dim = embedding_dim
        self.emb_shape = [self.embedding_size, self.embedding_dim]
        self.hidden_size = hidden_size
        self.user_encoder = GrnnEncoder(hidden_size=hidden_size)
        self.item_encoder = BowEncoder()
        self.pairwise_hinge_loss = PairwiseHingeLoss()

    def get_correct(self, x, y):
        less = tensor.cast(cf.less_than(x, y), dtype='float32')
        correct = nn.reduce_sum(less)
        return correct

    def train(self):
        user_data = io.data(name="user", shape=[1], dtype="int64", lod_level=1)
        pos_item_data = io.data(
            name="p_item", shape=[1], dtype="int64", lod_level=1)
        neg_item_data = io.data(
            name="n_item", shape=[1], dtype="int64", lod_level=1)
        user_emb = nn.embedding(
            input=user_data, size=self.emb_shape, param_attr="emb.item")
        pos_item_emb = nn.embedding(
            input=pos_item_data, size=self.emb_shape, param_attr="emb.item")
        neg_item_emb = nn.embedding(
            input=neg_item_data, size=self.emb_shape, param_attr="emb.item")
        user_enc = self.user_encoder.forward(user_emb)
        pos_item_enc = self.item_encoder.forward(pos_item_emb)
        neg_item_enc = self.item_encoder.forward(neg_item_emb)
        user_hid = nn.fc(input=user_enc,
                         size=self.hidden_size,
                         param_attr='user.w',
                         bias_attr="user.b")
        pos_item_hid = nn.fc(input=pos_item_enc,
                             size=self.hidden_size,
                             param_attr='item.w',
                             bias_attr="item.b")
        neg_item_hid = nn.fc(input=neg_item_enc,
                             size=self.hidden_size,
                             param_attr='item.w',
                             bias_attr="item.b")
        cos_pos = nn.cos_sim(user_hid, pos_item_hid)
        cos_neg = nn.cos_sim(user_hid, neg_item_hid)
        hinge_loss = self.pairwise_hinge_loss.forward(cos_pos, cos_neg)
        avg_cost = nn.mean(hinge_loss)
        correct = self.get_correct(cos_neg, cos_pos)

        return [user_data, pos_item_data,
                neg_item_data], cos_pos, avg_cost, correct
