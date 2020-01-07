# -*- coding: utf-8 -*-
#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
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

import paddle.fluid as fluid
import numpy as np
from paddle.fluid import ParamAttr
from paddle.fluid.dygraph import to_variable
from paddle.fluid.dygraph.nn import Embedding, Linear
from rnn import BasicLSTMUnit
import numpy as np

INF = 1. * 1e5
alpha = 0.6
uniform_initializer = lambda x: fluid.initializer.UniformInitializer(low=-x, high=x)
zero_constant = fluid.initializer.Constant(0.0)

class BaseModel(fluid.dygraph.Layer):
    def __init__(self,
                 hidden_size,
                 src_vocab_size,
                 tar_vocab_size,
                 batch_size,
                 num_layers=1,
                 init_scale=0.1,
                 dropout=None,
                 beam_size=1,
                 beam_start_token=1,
                 beam_end_token=2,
                 beam_max_step_num=100,
                 mode='train'):
        super(BaseModel, self).__init__()
        self.hidden_size = hidden_size
        self.src_vocab_size = src_vocab_size
        self.tar_vocab_size = tar_vocab_size
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.init_scale = init_scale
        self.dropout = dropout
        self.beam_size = beam_size
        self.beam_start_token = beam_start_token
        self.beam_end_token = beam_end_token
        self.beam_max_step_num = beam_max_step_num
        self.mode = mode
        self.kinf = 1e9

        param_attr = ParamAttr(initializer=uniform_initializer(self.init_scale))
        bias_attr = ParamAttr(initializer=zero_constant)
        forget_bias = 1.0

        self.src_embeder = Embedding(
            size=[self.src_vocab_size, self.hidden_size],
            param_attr=fluid.ParamAttr(
                initializer=uniform_initializer(init_scale)))

        self.tar_embeder = Embedding(
            size=[self.tar_vocab_size, self.hidden_size],
            is_sparse=False,
            param_attr=fluid.ParamAttr(
                initializer=uniform_initializer(init_scale)))

        self.enc_units = []
        for i in range(num_layers):
            self.enc_units.append(
                self.add_sublayer("enc_units_%d" % i,
                    BasicLSTMUnit(
                    hidden_size=self.hidden_size, 
                    input_size=self.hidden_size,
                    param_attr=param_attr, 
                    bias_attr=bias_attr, 
                    forget_bias=forget_bias)))

        self.dec_units = []
        for i in range(num_layers):
            self.dec_units.append(
                self.add_sublayer("dec_units_%d" % i,
                    BasicLSTMUnit(
                    hidden_size=self.hidden_size, 
                    input_size=self.hidden_size,
                    param_attr=param_attr, 
                    bias_attr=bias_attr, 
                    forget_bias=forget_bias)))
        
        self.fc = fluid.dygraph.nn.Linear(self.hidden_size,
                self.tar_vocab_size,
                param_attr=param_attr,
                bias_attr=False)

    def _transpose_batch_time(self, x):
        return fluid.layers.transpose(x, [1, 0] + list(range(2, len(x.shape))))

    def _merge_batch_beams(self, x):
        return fluid.layers.reshape(x, shape=(-1,x.shape[2]))

    def _split_batch_beams(self, x):
        return fluid.layers.reshape(x, shape=(-1, self.beam_size, x.shape[1]))

    def _expand_to_beam_size(self, x):
        x = fluid.layers.unsqueeze(x, [1])
        expand_times = [1] * len(x.shape)
        expand_times[1] = self.beam_size
        x = fluid.layers.expand(x, expand_times)
        return x

    def _real_state(self, state, new_state, step_mask):
        new_state = fluid.layers.elementwise_mul(new_state, step_mask, axis=0) - \
                    fluid.layers.elementwise_mul(state, (step_mask - 1), axis=0)
        return new_state

    def _gather(self, x, indices, batch_pos):
        topk_coordinates = fluid.layers.stack([batch_pos, indices], axis=2)
        return fluid.layers.gather_nd(x, topk_coordinates)

    def forward(self, inputs):
        #inputs[0] = np.expand_dims(inputs[0], axis=-1)
        #inputs[1] = np.expand_dims(inputs[1], axis=-1)
        inputs = [fluid.dygraph.to_variable(np_inp) for np_inp in inputs]
        src, tar, label, src_sequence_length, tar_sequence_length = inputs
        if src.shape[0] < self.batch_size:
            self.batch_size = src.shape[0]
        src_emb = self.src_embeder(self._transpose_batch_time(src))

        enc_hidden = to_variable(np.zeros((self.num_layers, self.batch_size, self.hidden_size), dtype='float32'))
        enc_cell = to_variable(np.zeros((self.num_layers, self.batch_size, self.hidden_size), dtype='float32'))

        max_seq_len = src_emb.shape[0]
        enc_len_mask = fluid.layers.sequence_mask(src_sequence_length, maxlen=max_seq_len, dtype="float32")
        enc_len_mask = fluid.layers.transpose(enc_len_mask, [1, 0])
        enc_states = [[enc_hidden, enc_cell]]
        for l in range(max_seq_len):
            step_input = src_emb[l]
            step_mask = enc_len_mask[l]
            enc_hidden, enc_cell = enc_states[l]
            new_enc_hidden, new_enc_cell = [], []
            for i in range(self.num_layers):
                new_hidden, new_cell = self.enc_units[i](step_input, enc_hidden[i], enc_cell[i])
                new_enc_hidden.append(new_hidden)
                new_enc_cell.append(new_cell)
                if self.dropout != None and self.dropout > 0.0:
                    step_input = fluid.layers.dropout(
                        new_hidden,
                        dropout_prob=self.dropout,
                        dropout_implementation='upscale_in_train')
                else:
                    step_input = new_hidden
            new_enc_hidden = [self._real_state(enc_hidden[i], new_enc_hidden[i], step_mask) for i in range(self.num_layers)]
            new_enc_cell = [self._real_state(enc_cell[i], new_enc_cell[i], step_mask) for i in range(self.num_layers)]
            enc_states.append([new_enc_hidden, new_enc_cell])
        
        if self.mode in ['train', 'eval']:
            dec_hidden, dec_cell = enc_states[-1]
            tar_emb = self.tar_embeder(self._transpose_batch_time(tar))
            max_seq_len = tar_emb.shape[0]
            dec_output = []

            for step_idx in range(max_seq_len):
                step_input = tar_emb[step_idx]
                new_dec_hidden, new_dec_cell = [], []
                for i in range(self.num_layers):
                    new_hidden, new_cell = self.dec_units[i](step_input, dec_hidden[i], dec_cell[i])
                    new_dec_hidden.append(new_hidden)
                    new_dec_cell.append(new_cell)
                    if self.dropout != None and self.dropout > 0.0:
                        step_input = fluid.layers.dropout(
                            new_hidden,
                            dropout_prob=self.dropout,
                            dropout_implementation='upscale_in_train')
                    else:
                        step_input = new_hidden
                dec_output.append(step_input)
                dec_hidden, dec_cell = new_dec_hidden, new_dec_cell

            dec_output = fluid.layers.stack(dec_output)
            dec_output = self.fc(self._transpose_batch_time(dec_output))
        
            loss = fluid.layers.softmax_with_cross_entropy(
            logits=dec_output, label=label, soft_label=False)
            loss = fluid.layers.squeeze(loss, axes=[2])
            max_tar_seq_len = fluid.layers.shape(tar)[1]
            tar_mask = fluid.layers.sequence_mask(
                tar_sequence_length, maxlen=max_tar_seq_len, dtype='float32')
            loss = loss * tar_mask
            loss = fluid.layers.reduce_mean(loss, dim=[0])
            loss = fluid.layers.reduce_sum(loss)
            return loss
        elif self.mode in ['beam_search']:
            batch_beam_shape = (self.batch_size, self.beam_size)
            #batch_beam_shape_1 = (self.batch_size, self.beam_size, 1)
            vocab_size_tensor = to_variable(np.full((1), self.tar_vocab_size))
            start_token_tensor = to_variable(np.full(batch_beam_shape, self.beam_start_token, dtype='int64')) # remove last dim 1 in v1.7
            end_token_tensor = to_variable(np.full(batch_beam_shape, self.beam_end_token, dtype='int64'))
            step_input = self.tar_embeder(start_token_tensor)
            beam_finished = to_variable(np.full(batch_beam_shape, 0, dtype='float32'))
            beam_state_log_probs = to_variable(np.array([[0.] + [-self.kinf] * (self.beam_size - 1)], dtype="float32"))
            beam_state_log_probs = fluid.layers.expand(beam_state_log_probs, [self.batch_size, 1])
            
            dec_hidden, dec_cell = enc_states[-1]
            dec_hidden = [self._expand_to_beam_size(state) for state in dec_hidden]
            dec_cell = [self._expand_to_beam_size(state) for state in dec_cell]
            
            batch_pos = fluid.layers.expand(
                fluid.layers.unsqueeze(to_variable(np.arange(0, self.batch_size, 1, dtype="int64")), [1]),
                [1, self.beam_size])
            predicted_ids = []
            parent_ids = []

            for step_idx in range(self.beam_max_step_num):
                if fluid.layers.reduce_sum(1 - beam_finished).numpy()[0] == 0:
                    break
                step_input = self._merge_batch_beams(step_input)
                new_dec_hidden, new_dec_cell = [], []
                dec_hidden = [self._merge_batch_beams(state) for state in dec_hidden]
                dec_cell = [self._merge_batch_beams(state) for state in dec_cell]

                for i in range(self.num_layers):
                    new_hidden, new_cell = self.dec_units[i](step_input, dec_hidden[i], dec_cell[i])
                    new_dec_hidden.append(new_hidden)
                    new_dec_cell.append(new_cell)
                    if self.dropout != None and self.dropout > 0.0:
                        step_input = fluid.layers.dropout(
                            new_hidden,
                            dropout_prob=self.dropout,
                            dropout_implementation='upscale_in_train')
                    else:
                        step_input = new_hidden
                cell_outputs = self._split_batch_beams(step_input)
                cell_outputs = self.fc(cell_outputs) 
                # Beam_search_step:
                step_log_probs = fluid.layers.log(fluid.layers.softmax(cell_outputs))
                noend_array = [-self.kinf] * self.tar_vocab_size
                noend_array[self.beam_end_token] = 0 # [-kinf, -kinf, ..., 0, -kinf, ...]
                noend_mask_tensor = to_variable(np.array(noend_array,dtype='float32'))
                # set finished position to one-hot probability of <eos>
                step_log_probs = fluid.layers.elementwise_mul(
                        fluid.layers.expand(fluid.layers.unsqueeze(beam_finished, [2]), [1, 1, self.tar_vocab_size]),
                    noend_mask_tensor, axis=-1) - \
                    fluid.layers.elementwise_mul(step_log_probs, (beam_finished - 1), axis=0)
                log_probs = fluid.layers.elementwise_add(
                    x=step_log_probs, y=beam_state_log_probs, axis=0)
                scores = fluid.layers.reshape(log_probs, [-1, self.beam_size * self.tar_vocab_size])
                topk_scores, topk_indices = fluid.layers.topk(input=scores, k=self.beam_size)
                beam_indices = fluid.layers.elementwise_floordiv(topk_indices, vocab_size_tensor) # in which beam
                token_indices = fluid.layers.elementwise_mod(topk_indices, vocab_size_tensor) # position in beam
                next_log_probs = self._gather(scores, topk_indices, batch_pos) # 

                new_dec_hidden = [self._split_batch_beams(state) for state in new_dec_hidden]
                new_dec_cell = [self._split_batch_beams(state) for state in new_dec_cell]
                new_dec_hidden = [self._gather(x, beam_indices, batch_pos) for x in new_dec_hidden]
                new_dec_cell = [self._gather(x, beam_indices, batch_pos) for x in new_dec_cell]
                
                next_finished = self._gather(beam_finished, beam_indices, batch_pos)              
                next_finished = fluid.layers.cast(next_finished, "bool")
                next_finished = fluid.layers.logical_or(next_finished, fluid.layers.equal(token_indices, end_token_tensor))
                next_finished = fluid.layers.cast(next_finished, "float32")
                # prepare for next step
                dec_hidden, dec_cell = new_dec_hidden, new_dec_cell
                beam_finished = next_finished
                beam_state_log_probs = next_log_probs
                step_input = self.tar_embeder(fluid.layers.unsqueeze(token_indices, 2)) # remove unsqueeze in v1.7
                predicted_ids.append(token_indices)
                parent_ids.append(beam_indices)

            predicted_ids = fluid.layers.stack(predicted_ids)
            parent_ids = fluid.layers.stack(parent_ids)
            predicted_ids = fluid.layers.gather_tree(predicted_ids, parent_ids)
            predicted_ids = self._transpose_batch_time(predicted_ids)
            return predicted_ids
        else:
            print("not support mode ", self.mode)
            raise Exception("not support mode: " + self.mode)