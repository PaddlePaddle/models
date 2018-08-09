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
from paddle.fluid.contrib.decoder.beam_search_decoder import *


def seq_to_seq_net(embedding_dim, encoder_size, decoder_size, source_dict_dim,
                   target_dict_dim, is_generating, beam_size, max_length):
    def encoder():
        # Encoder implementation of RNN translation
        src_word = layers.data(
            name="src_word", shape=[1], dtype='int64', lod_level=1)
        src_embedding = layers.embedding(
            input=src_word,
            size=[source_dict_dim, embedding_dim],
            dtype='float32',
            is_sparse=True)

        fc1 = layers.fc(input=src_embedding, size=encoder_size * 4, act='tanh')
        lstm_hidden0, lstm_0 = layers.dynamic_lstm(
            input=fc1, size=encoder_size * 4)
        encoder_out = layers.sequence_last_step(input=lstm_hidden0)
        return encoder_out

    def decoder_state_cell(context):
        # Decoder state cell, specifies the hidden state variable and its updater
        h = InitState(init=context, need_reorder=True)
        state_cell = StateCell(
            inputs={'x': None}, states={'h': h}, out_state='h')

        @state_cell.state_updater
        def updater(state_cell):
            current_word = state_cell.get_input('x')
            prev_h = state_cell.get_state('h')
            # make sure lod of h heritted from prev_h
            h = layers.fc(input=[prev_h, current_word],
                          size=decoder_size,
                          act='tanh')
            state_cell.set_state('h', h)

        return state_cell

    def decoder_train(state_cell):
        # Decoder for training implementation of RNN translation
        trg_word = layers.data(
            name="target_word", shape=[1], dtype='int64', lod_level=1)
        trg_embedding = layers.embedding(
            input=trg_word,
            size=[target_dict_dim, embedding_dim],
            dtype='float32',
            is_sparse=True)

        # A training decoder
        decoder = TrainingDecoder(state_cell)

        # Define the computation in each RNN step done by decoder
        with decoder.block():
            current_word = decoder.step_input(trg_embedding)
            decoder.state_cell.compute_state(inputs={'x': current_word})
            current_score = layers.fc(input=decoder.state_cell.get_state('h'),
                                      size=target_dict_dim,
                                      act='softmax')
            decoder.state_cell.update_states()
            decoder.output(current_score)

        return decoder()

    def decoder_infer(state_cell):
        # Decoder for inference implementation
        init_ids = layers.data(
            name="init_ids", shape=[1], dtype="int64", lod_level=2)
        init_scores = layers.data(
            name="init_scores", shape=[1], dtype="float32", lod_level=2)

        # A beam search decoder for inference
        decoder = BeamSearchDecoder(
            state_cell=state_cell,
            init_ids=init_ids,
            init_scores=init_scores,
            target_dict_dim=target_dict_dim,
            word_dim=embedding_dim,
            input_var_dict={},
            topk_size=50,
            sparse_emb=True,
            max_len=max_length,
            beam_size=beam_size,
            end_id=1,
            name=None)
        decoder.decode()
        translation_ids, translation_scores = decoder()

        return translation_ids, translation_scores

    context = encoder()
    state_cell = decoder_state_cell(context)

    if not is_generating:
        label = layers.data(
            name="target_next_word", shape=[1], dtype='int64', lod_level=1)

        rnn_out = decoder_train(state_cell)

        cost = layers.cross_entropy(input=rnn_out, label=label)
        avg_cost = layers.mean(x=cost)

        feeding_list = ['src_word', 'target_word', 'target_next_word']
        return avg_cost, feeding_list
    else:
        translation_ids, translation_scores = decoder_infer(state_cell)
        feeding_list = ['src_word']
        return translation_ids, translation_scores, feeding_list
