#  Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

import paddle
import math
import numpy as np
import paddle.fluid as fluid
import paddle.fluid.layers as layers


def network(items_num, hidden_size, step):
    stdv = 1.0 / math.sqrt(hidden_size)

    items = layers.data(
        name="items",
        shape=[1, 1],
        dtype="int64") #[batch_size, uniq_max, 1]
    seq_index = layers.data(
        name="seq_index",
        shape=[1],
        dtype="int32") #[batch_size, seq_max]
    last_index = layers.data(
        name="last_index",
        shape=[1],
        dtype="int32") #[batch_size, 1]
    adj_in = layers.data(
        name="adj_in",
        shape=[1,1],
        dtype="float32") #[batch_size, seq_max, seq_max]
    adj_out = layers.data(
        name="adj_out",
        shape=[1,1],
        dtype="float32") #[batch_size, seq_max, seq_max]
    mask = layers.data(
        name="mask",
        shape=[1, 1],
        dtype="float32") #[batch_size, seq_max, 1]
    label = layers.data(
        name="label",
        shape=[1],
        dtype="int64") #[batch_size, 1]

    datas = [items, seq_index, last_index, adj_in, adj_out, mask, label]
    py_reader = fluid.layers.create_py_reader_by_data(
                    capacity=256, feed_list=datas, name='py_reader', use_double_buffer=True)
    feed_datas = fluid.layers.read_file(py_reader)
    items, seq_index, last_index, adj_in, adj_out, mask, label = feed_datas

    items_emb = layers.embedding(
        input=items,
        param_attr=fluid.ParamAttr(
            name="emb",
            initializer=fluid.initializer.Uniform(
                low=-stdv, high=stdv)),
        size=[items_num, hidden_size])  #[batch_size, uniq_max, h]
    items_emb_shape = layers.shape(items_emb)

    pre_state = items_emb
    for i in range(step):
        pre_state = layers.reshape(
            x=pre_state, shape=[-1, 1, hidden_size], actual_shape=items_emb_shape)
        state_in = layers.fc(
            input=pre_state,
            name="state_in",
            size=hidden_size,
            act=None,
            num_flatten_dims=2,
            param_attr=fluid.ParamAttr(initializer=fluid.initializer.Uniform(
                low=-stdv, high=stdv)),
            bias_attr=fluid.ParamAttr(initializer=fluid.initializer.Uniform(
                low=-stdv, high=stdv)))  #[batch_size, uniq_max, h]
        state_out = layers.fc(
            input=pre_state,
            name="state_out",
            size=hidden_size,
            act=None,
            num_flatten_dims=2,
            param_attr=fluid.ParamAttr(initializer=fluid.initializer.Uniform(
                low=-stdv, high=stdv)),
            bias_attr=fluid.ParamAttr(initializer=fluid.initializer.Uniform(
                low=-stdv, high=stdv)))  #[batch_size, uniq_max, h]

        state_adj_in = layers.matmul(adj_in, state_in)  #[batch_size, uniq_max, h]
        state_adj_out = layers.matmul(adj_out, state_out)   #[batch_size, uniq_max, h]

        gru_input = layers.concat([state_adj_in, state_adj_out], axis=2)

        gru_input = layers.reshape(x=gru_input, shape=[-1, hidden_size * 2])
        gru_fc = layers.fc(
            input=gru_input,
            name="gru_fc",
            size=3 * hidden_size,
            bias_attr=False)
        pre_state, _, _ = fluid.layers.gru_unit(
            input=gru_fc,
            hidden=layers.reshape(
                x=pre_state, shape=[-1, hidden_size]),
            size=3 * hidden_size)

    final_state = pre_state #[batch_size * uniq_max, h]

    seq_origin_shape = layers.assign(np.array([0,0,hidden_size-1]).astype("int32"))
    seq_origin_shape += layers.shape(layers.unsqueeze(seq_index,[2])) #value: [batch_size, seq_max, h]
    seq_origin_shape.stop_gradient = True

    seq_index = layers.reshape(seq_index, shape=[-1])
    seq = layers.gather(final_state, seq_index)  #[batch_size * seq_max, h]
    last = layers.gather(final_state, last_index)  #[batch_size, h]

    seq = layers.reshape(
        seq, shape=[-1, 1, hidden_size], actual_shape=seq_origin_shape)  #[batch_size, seq_max, h]
    last = layers.reshape(
        last, shape=[-1, hidden_size])  #[batch_size, h]

    seq_fc = layers.fc(
        input=seq,
        name="seq_fc",
        size=hidden_size,
        bias_attr=False,
        act=None,
        num_flatten_dims=2,
        param_attr=fluid.ParamAttr(
            initializer=fluid.initializer.Uniform(
            low=-stdv, high=stdv)))  #[batch_size, seq_max, h]
    last_fc = layers.fc(
        input=last,
        name="last_fc",
        size=hidden_size,
        bias_attr=False,
        act=None,
        num_flatten_dims=1,
        param_attr=fluid.ParamAttr(
            initializer=fluid.initializer.Uniform(
            low=-stdv, high=stdv)))  #[bathc_size, h]

    seq_fc_t = layers.transpose(
        seq_fc, perm=[1, 0, 2])  #[seq_max, batch_size, h]
    add = layers.elementwise_add(
        seq_fc_t, last_fc)  #[seq_max, batch_size, h]
    b = layers.create_parameter(
        shape=[hidden_size],
        dtype='float32',
        default_initializer=fluid.initializer.Constant(value=0.0))  #[h]
    add = layers.elementwise_add(add, b)  #[seq_max, batch_size, h]

    add_sigmoid = layers.sigmoid(add) #[seq_max, batch_size, h] 
    add_sigmoid = layers.transpose(
        add_sigmoid, perm=[1, 0, 2])  #[batch_size, seq_max, h]

    weight = layers.fc(
        input=add_sigmoid,
        name="weight_fc",
        size=1,
        act=None,
        num_flatten_dims=2,
        bias_attr=False,
        param_attr=fluid.ParamAttr(
            initializer=fluid.initializer.Uniform(
                low=-stdv, high=stdv)))  #[batch_size, seq_max, 1]
    weight *= mask
    weight_mask = layers.elementwise_mul(seq, weight, axis=0) #[batch_size, seq_max, h]
    global_attention = layers.reduce_sum(weight_mask, dim=1) #[batch_size, h]

    final_attention = layers.concat(
        [global_attention, last], axis=1)  #[batch_size, 2*h]
    final_attention_fc = layers.fc(
        input=final_attention,
        name="final_attention_fc",
        size=hidden_size,
        bias_attr=False,
        act=None,
        param_attr=fluid.ParamAttr(initializer=fluid.initializer.Uniform(
            low=-stdv, high=stdv)))  #[batch_size, h]

    all_vocab = layers.create_global_var(
        shape=[items_num - 1, 1],
        value=0,
        dtype="int64",
        persistable=True,
        name="all_vocab")

    all_emb = layers.embedding(
        input=all_vocab,
        param_attr=fluid.ParamAttr(
            name="emb",
            initializer=fluid.initializer.Uniform(
                low=-stdv, high=stdv)),
        size=[items_num, hidden_size])  #[all_vocab, h]

    logits = layers.matmul(
        x=final_attention_fc, y=all_emb,
        transpose_y=True)  #[batch_size, all_vocab]
    softmax = layers.softmax_with_cross_entropy(
        logits=logits, label=label)  #[batch_size, 1]
    loss = layers.reduce_mean(softmax)  # [1]
    acc = layers.accuracy(input=logits, label=label, k=20)
    return loss, acc, py_reader, feed_datas
