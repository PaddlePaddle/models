#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle.fluid.layers as layers
import paddle.fluid as fluid
from paddle.fluid.layers.control_flow import StaticRNN as PaddingRNN
import numpy as np
from paddle.fluid import ParamAttr
from paddle.fluid.contrib.layers import basic_lstm, BasicLSTMUnit

INF = 1. * 1e5
alpha = 0.6


class BaseModel(object):
    def __init__(self,
                 hidden_size,
                 src_vocab_size,
                 tar_vocab_size,
                 batch_size,
                 num_layers=1,
                 init_scale=0.1,
                 dropout=None,
                 batch_first=True):

        self.hidden_size = hidden_size
        self.src_vocab_size = src_vocab_size
        self.tar_vocab_size = tar_vocab_size
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.init_scale = init_scale
        self.dropout = dropout
        self.batch_first = batch_first

    def _build_data(self):
        self.src = layers.data(name="src", shape=[-1, 1, 1], dtype='int64')
        self.src_sequence_length = layers.data(
            name="src_sequence_length", shape=[-1], dtype='int32')

        self.tar = layers.data(name="tar", shape=[-1, 1, 1], dtype='int64')
        self.tar_sequence_length = layers.data(
            name="tar_sequence_length", shape=[-1], dtype='int32')
        self.label = layers.data(name="label", shape=[-1, 1, 1], dtype='int64')

    def _emebdding(self):
        self.src_emb = layers.embedding(
            input=self.src,
            size=[self.src_vocab_size, self.hidden_size],
            dtype='float32',
            is_sparse=False,
            param_attr=fluid.ParamAttr(
                name='source_embedding',
                initializer=fluid.initializer.UniformInitializer(
                    low=-self.init_scale, high=self.init_scale)))
        self.tar_emb = layers.embedding(
            input=self.tar,
            size=[self.tar_vocab_size, self.hidden_size],
            dtype='float32',
            is_sparse=False,
            param_attr=fluid.ParamAttr(
                name='target_embedding',
                initializer=fluid.initializer.UniformInitializer(
                    low=-self.init_scale, high=self.init_scale)))

    def _build_encoder(self):
        self.enc_output, enc_last_hidden, enc_last_cell = basic_lstm( self.src_emb, None, None, self.hidden_size, num_layers=self.num_layers, batch_first=self.batch_first, \
                dropout_prob=self.dropout, \
                param_attr = ParamAttr( initializer=fluid.initializer.UniformInitializer(low=-self.init_scale, high=self.init_scale) ), \
                bias_attr = ParamAttr( initializer = fluid.initializer.Constant(0.0) ), \
                sequence_length=self.src_sequence_length)

        return self.enc_output, enc_last_hidden, enc_last_cell

    def _build_decoder(self,
                       enc_last_hidden,
                       enc_last_cell,
                       mode='train',
                       beam_size=10):
        softmax_weight = layers.create_parameter([self.hidden_size, self.tar_vocab_size], dtype="float32", name="softmax_weight", \
                    default_initializer=fluid.initializer.UniformInitializer(low=-self.init_scale, high=self.init_scale))
        if mode == 'train':
            dec_output, dec_last_hidden, dec_last_cell = basic_lstm( self.tar_emb, enc_last_hidden, enc_last_cell, \
                    self.hidden_size, num_layers=self.num_layers, \
                    batch_first=self.batch_first, \
                    dropout_prob=self.dropout, \
                    param_attr = ParamAttr( initializer=fluid.initializer.UniformInitializer(low=-self.init_scale, high=self.init_scale) ), \
                    bias_attr = ParamAttr( initializer = fluid.initializer.Constant(0.0) ))

            dec_output = layers.matmul(dec_output, softmax_weight)

            return dec_output
        elif mode == 'beam_search' or mode == 'greedy_search':
            dec_unit_list = []
            name = 'basic_lstm'
            for i in range(self.num_layers):
                new_name = name + "_layers_" + str(i)
                dec_unit_list.append(
                    BasicLSTMUnit(
                        new_name, self.hidden_size, dtype='float32'))

            def decoder_step(current_in, pre_hidden_array, pre_cell_array):
                new_hidden_array = []
                new_cell_array = []

                step_in = current_in
                for i in range(self.num_layers):
                    pre_hidden = pre_hidden_array[i]
                    pre_cell = pre_cell_array[i]

                    new_hidden, new_cell = dec_unit_list[i](step_in, pre_hidden,
                                                            pre_cell)

                    new_hidden_array.append(new_hidden)
                    new_cell_array.append(new_cell)

                    step_in = new_hidden

                return step_in, new_hidden_array, new_cell_array

            if mode == 'beam_search':
                max_src_seq_len = layers.shape(self.src)[1]
                max_length = max_src_seq_len * 2
                #max_length = layers.fill_constant( [1], dtype='int32', value = 10)
                pre_ids = layers.fill_constant([1, 1], dtype='int64', value=1)
                full_ids = layers.fill_constant([1, 1], dtype='int64', value=1)

                score = layers.fill_constant([1], dtype='float32', value=0.0)

                #eos_ids = layers.fill_constant( [1, 1], dtype='int64', value=2)

                pre_hidden_array = []
                pre_cell_array = []
                pre_feed = layers.fill_constant(
                    [beam_size, self.hidden_size], dtype='float32', value=0)
                for i in range(self.num_layers):
                    pre_hidden_array.append(
                        layers.expand(enc_last_hidden[i], [beam_size, 1]))
                    pre_cell_array.append(
                        layers.expand(enc_last_cell[i], [beam_size, 1]))

                eos_ids = layers.fill_constant(
                    [beam_size], dtype='int64', value=2)
                init_score = np.zeros((beam_size)).astype('float32')
                init_score[1:] = -INF
                pre_score = layers.assign(init_score)
                #pre_score = layers.fill_constant( [1,], dtype='float32', value= 0.0) 
                tokens = layers.fill_constant(
                    [beam_size, 1], dtype='int64', value=1)

                enc_memory = layers.expand(self.enc_output, [beam_size, 1, 1])

                pre_tokens = layers.fill_constant(
                    [beam_size, 1], dtype='int64', value=1)

                finished_seq = layers.fill_constant(
                    [beam_size, 1], dtype='int64', value=0)
                finished_scores = layers.fill_constant(
                    [beam_size], dtype='float32', value=-INF)
                finished_flag = layers.fill_constant(
                    [beam_size], dtype='float32', value=0.0)

                step_idx = layers.fill_constant(
                    shape=[1], dtype='int32', value=0)
                cond = layers.less_than(
                    x=step_idx, y=max_length)  # default force_cpu=True

                parent_idx = layers.fill_constant([1], dtype='int32', value=0)
                while_op = layers.While(cond)

                def compute_topk_scores_and_seq(sequences,
                                                scores,
                                                scores_to_gather,
                                                flags,
                                                beam_size,
                                                select_beam=None,
                                                generate_id=None):
                    scores = layers.reshape(scores, shape=[1, -1])
                    _, topk_indexs = layers.topk(scores, k=beam_size)

                    topk_indexs = layers.reshape(topk_indexs, shape=[-1])

                    # gather result

                    top_seq = layers.gather(sequences, topk_indexs)
                    topk_flags = layers.gather(flags, topk_indexs)
                    topk_gather_scores = layers.gather(scores_to_gather,
                                                       topk_indexs)

                    if select_beam:
                        topk_beam = layers.gather(select_beam, topk_indexs)
                    else:
                        topk_beam = select_beam

                    if generate_id:
                        topk_id = layers.gather(generate_id, topk_indexs)
                    else:
                        topk_id = generate_id
                    return top_seq, topk_gather_scores, topk_flags, topk_beam, topk_id

                def grow_alive(curr_seq, curr_scores, curr_log_probs,
                               curr_finished, select_beam, generate_id):
                    curr_scores += curr_finished * -INF
                    return compute_topk_scores_and_seq(
                        curr_seq,
                        curr_scores,
                        curr_log_probs,
                        curr_finished,
                        beam_size,
                        select_beam,
                        generate_id=generate_id)

                def grow_finished(finished_seq, finished_scores, finished_flag,
                                  curr_seq, curr_scores, curr_finished):
                    finished_seq = layers.concat(
                        [
                            finished_seq, layers.fill_constant(
                                [beam_size, 1], dtype='int64', value=1)
                        ],
                        axis=1)
                    curr_scores += (1.0 - curr_finished) * -INF
                    #layers.Print( curr_scores, message="curr scores")
                    curr_finished_seq = layers.concat(
                        [finished_seq, curr_seq], axis=0)
                    curr_finished_scores = layers.concat(
                        [finished_scores, curr_scores], axis=0)
                    curr_finished_flags = layers.concat(
                        [finished_flag, curr_finished], axis=0)

                    return compute_topk_scores_and_seq(
                        curr_finished_seq, curr_finished_scores,
                        curr_finished_scores, curr_finished_flags, beam_size)

                def is_finished(alive_log_prob, finished_scores,
                                finished_in_finished):

                    max_out_len = 200
                    max_length_penalty = layers.pow(layers.fill_constant(
                        [1], dtype='float32', value=(
                            (5.0 + max_out_len) / 6.0)),
                                                    alpha)

                    lower_bound_alive_score = layers.slice(
                        alive_log_prob, starts=[0], ends=[1],
                        axes=[0]) / max_length_penalty

                    lowest_score_of_fininshed_in_finished = finished_scores * finished_in_finished
                    lowest_score_of_fininshed_in_finished += (
                        1.0 - finished_in_finished) * -INF
                    lowest_score_of_fininshed_in_finished = layers.reduce_min(
                        lowest_score_of_fininshed_in_finished)

                    met = layers.less_than(
                        lower_bound_alive_score,
                        lowest_score_of_fininshed_in_finished)
                    met = layers.cast(met, 'float32')
                    bound_is_met = layers.reduce_sum(met)

                    finished_eos_num = layers.reduce_sum(finished_in_finished)

                    finish_cond = layers.less_than(
                        finished_eos_num,
                        layers.fill_constant(
                            [1], dtype='float32', value=beam_size))

                    return finish_cond

                def grow_top_k(step_idx, alive_seq, alive_log_prob, parant_idx):
                    pre_ids = alive_seq

                    dec_step_emb = layers.embedding(
                        input=pre_ids,
                        size=[self.tar_vocab_size, self.hidden_size],
                        dtype='float32',
                        is_sparse=False,
                        param_attr=fluid.ParamAttr(
                            name='target_embedding',
                            initializer=fluid.initializer.UniformInitializer(
                                low=-self.init_scale, high=self.init_scale)))

                    dec_att_out, new_hidden_array, new_cell_array = decoder_step(
                        dec_step_emb, pre_hidden_array, pre_cell_array)

                    projection = layers.matmul(dec_att_out, softmax_weight)

                    logits = layers.softmax(projection)
                    current_log = layers.elementwise_add(
                        x=layers.log(logits), y=alive_log_prob, axis=0)
                    base_1 = layers.cast(step_idx, 'float32') + 6.0
                    base_1 /= 6.0
                    length_penalty = layers.pow(base_1, alpha)

                    len_pen = layers.pow(((
                        5. + layers.cast(step_idx + 1, 'float32')) / 6.), alpha)

                    current_log = layers.reshape(current_log, shape=[1, -1])

                    current_log = current_log / length_penalty
                    topk_scores, topk_indices = layers.topk(
                        input=current_log, k=beam_size)

                    topk_scores = layers.reshape(topk_scores, shape=[-1])

                    topk_log_probs = topk_scores * length_penalty

                    generate_id = layers.reshape(
                        topk_indices, shape=[-1]) % self.tar_vocab_size

                    selected_beam = layers.reshape(
                        topk_indices, shape=[-1]) // self.tar_vocab_size

                    topk_finished = layers.equal(generate_id, eos_ids)

                    topk_finished = layers.cast(topk_finished, 'float32')

                    generate_id = layers.reshape(generate_id, shape=[-1, 1])

                    pre_tokens_list = layers.gather(tokens, selected_beam)

                    full_tokens_list = layers.concat(
                        [pre_tokens_list, generate_id], axis=1)


                    return full_tokens_list, topk_log_probs, topk_scores, topk_finished, selected_beam, generate_id, \
                            dec_att_out, new_hidden_array, new_cell_array

                with while_op.block():
                    topk_seq, topk_log_probs, topk_scores, topk_finished, topk_beam, topk_generate_id, attention_out, new_hidden_array, new_cell_array = \
                        grow_top_k(  step_idx, pre_tokens, pre_score, parent_idx)
                    alive_seq, alive_log_prob, _, alive_beam, alive_id = grow_alive(
                        topk_seq, topk_scores, topk_log_probs, topk_finished,
                        topk_beam, topk_generate_id)

                    finished_seq_2, finished_scores_2, finished_flags_2, _, _ = grow_finished(
                        finished_seq, finished_scores, finished_flag, topk_seq,
                        topk_scores, topk_finished)

                    finished_cond = is_finished(
                        alive_log_prob, finished_scores_2, finished_flags_2)

                    layers.increment(x=step_idx, value=1.0, in_place=True)

                    layers.assign(alive_beam, parent_idx)
                    layers.assign(alive_id, pre_tokens)
                    layers.assign(alive_log_prob, pre_score)
                    layers.assign(alive_seq, tokens)
                    layers.assign(finished_seq_2, finished_seq)
                    layers.assign(finished_scores_2, finished_scores)
                    layers.assign(finished_flags_2, finished_flag)

                    # update init_hidden, init_cell, input_feed
                    new_feed = layers.gather(attention_out, parent_idx)
                    layers.assign(new_feed, pre_feed)
                    for i in range(self.num_layers):
                        new_hidden_var = layers.gather(new_hidden_array[i],
                                                       parent_idx)
                        layers.assign(new_hidden_var, pre_hidden_array[i])
                        new_cell_var = layers.gather(new_cell_array[i],
                                                     parent_idx)
                        layers.assign(new_cell_var, pre_cell_array[i])

                    length_cond = layers.less_than(x=step_idx, y=max_length)
                    layers.logical_and(x=length_cond, y=finished_cond, out=cond)

                tokens_with_eos = tokens

                all_seq = layers.concat([tokens_with_eos, finished_seq], axis=0)
                all_score = layers.concat([pre_score, finished_scores], axis=0)
                _, topk_index = layers.topk(all_score, k=beam_size)
                topk_index = layers.reshape(topk_index, shape=[-1])
                final_seq = layers.gather(all_seq, topk_index)
                final_score = layers.gather(all_score, topk_index)

                return final_seq
            elif mode == 'greedy_search':
                max_src_seq_len = layers.shape(self.src)[1]
                max_length = max_src_seq_len * 2
                #max_length = layers.fill_constant( [1], dtype='int32', value = 10)
                pre_ids = layers.fill_constant([1, 1], dtype='int64', value=1)
                full_ids = layers.fill_constant([1, 1], dtype='int64', value=1)

                score = layers.fill_constant([1], dtype='float32', value=0.0)

                eos_ids = layers.fill_constant([1, 1], dtype='int64', value=2)

                pre_hidden_array = []
                pre_cell_array = []
                pre_feed = layers.fill_constant(
                    [1, self.hidden_size], dtype='float32', value=0)
                for i in range(self.num_layers):
                    pre_hidden_array.append(enc_last_hidden[i])
                    pre_cell_array.append(enc_last_cell[i])
                    #pre_hidden_array.append( layers.fill_constant( [1, hidden_size], dtype='float32', value=0)  )
                    #pre_cell_array.append( layers.fill_constant( [1, hidden_size], dtype='float32', value=0) )

                step_idx = layers.fill_constant(
                    shape=[1], dtype='int32', value=0)
                cond = layers.less_than(
                    x=step_idx, y=max_length)  # default force_cpu=True
                while_op = layers.While(cond)

                with while_op.block():

                    dec_step_emb = layers.embedding(
                        input=pre_ids,
                        size=[self.tar_vocab_size, self.hidden_size],
                        dtype='float32',
                        is_sparse=False,
                        param_attr=fluid.ParamAttr(
                            name='target_embedding',
                            initializer=fluid.initializer.UniformInitializer(
                                low=-self.init_scale, high=self.init_scale)))

                    dec_att_out, new_hidden_array, new_cell_array = decoder_step(
                        dec_step_emb, pre_hidden_array, pre_cell_array)

                    projection = layers.matmul(dec_att_out, softmax_weight)

                    logits = layers.softmax(projection)
                    logits = layers.log(logits)

                    current_log = layers.elementwise_add(logits, score, axis=0)

                    topk_score, topk_indices = layers.topk(
                        input=current_log, k=1)

                    new_ids = layers.concat([full_ids, topk_indices])
                    layers.assign(new_ids, full_ids)
                    #layers.Print( full_ids, message="ful ids")
                    layers.assign(topk_score, score)
                    layers.assign(topk_indices, pre_ids)
                    layers.assign(dec_att_out, pre_feed)
                    for i in range(self.num_layers):
                        layers.assign(new_hidden_array[i], pre_hidden_array[i])
                        layers.assign(new_cell_array[i], pre_cell_array[i])

                    layers.increment(x=step_idx, value=1.0, in_place=True)

                    eos_met = layers.not_equal(topk_indices, eos_ids)
                    length_cond = layers.less_than(x=step_idx, y=max_length)
                    layers.logical_and(x=length_cond, y=eos_met, out=cond)

                return full_ids

            raise Exception("error")
        else:
            print("mode not supprt", mode)

    def _compute_loss(self, dec_output):
        loss = layers.softmax_with_cross_entropy(
            logits=dec_output, label=self.label, soft_label=False)

        loss = layers.reshape(loss, shape=[self.batch_size, -1])

        max_tar_seq_len = layers.shape(self.tar)[1]
        tar_mask = layers.sequence_mask(
            self.tar_sequence_length, maxlen=max_tar_seq_len, dtype='float32')
        loss = loss * tar_mask
        loss = layers.reduce_mean(loss, dim=[0])
        loss = layers.reduce_sum(loss)

        loss.permissions = True

        return loss

    def _beam_search(self, enc_last_hidden, enc_last_cell):
        pass

    def build_graph(self, mode='train', beam_size=10):
        if mode == 'train' or mode == 'eval':
            self._build_data()
            self._emebdding()
            enc_output, enc_last_hidden, enc_last_cell = self._build_encoder()
            dec_output = self._build_decoder(enc_last_hidden, enc_last_cell)

            loss = self._compute_loss(dec_output)
            return loss
        elif mode == "beam_search" or mode == 'greedy_search':
            self._build_data()
            self._emebdding()
            enc_output, enc_last_hidden, enc_last_cell = self._build_encoder()
            dec_output = self._build_decoder(
                enc_last_hidden, enc_last_cell, mode=mode, beam_size=beam_size)

            return dec_output
        else:
            print("not support mode ", mode)
            raise Exception("not support mode: " + mode)
