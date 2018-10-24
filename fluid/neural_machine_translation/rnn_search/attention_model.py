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

import paddle.fluid as fluid
from paddle.fluid.contrib.decoder.beam_search_decoder import *


def lstm_step(x_t, hidden_t_prev, cell_t_prev, size):
    def linear(inputs):
        return fluid.layers.fc(input=inputs, size=size, bias_attr=True)

    forget_gate = fluid.layers.sigmoid(x=linear([hidden_t_prev, x_t]))
    input_gate = fluid.layers.sigmoid(x=linear([hidden_t_prev, x_t]))
    output_gate = fluid.layers.sigmoid(x=linear([hidden_t_prev, x_t]))
    cell_tilde = fluid.layers.tanh(x=linear([hidden_t_prev, x_t]))

    cell_t = fluid.layers.sums(input=[
        fluid.layers.elementwise_mul(
            x=forget_gate, y=cell_t_prev), fluid.layers.elementwise_mul(
                x=input_gate, y=cell_tilde)
    ])

    hidden_t = fluid.layers.elementwise_mul(
        x=output_gate, y=fluid.layers.tanh(x=cell_t))

    return hidden_t, cell_t


def seq_to_seq_net(embedding_dim, encoder_size, decoder_size, source_dict_dim,
                   target_dict_dim, is_generating, beam_size, max_length):
    """Construct a seq2seq network."""

    def bi_lstm_encoder(input_seq, gate_size):
        # A bi-directional lstm encoder implementation.
        # Linear transformation part for input gate, output gate, forget gate
        # and cell activation vectors need be done outside of dynamic_lstm.
        # So the output size is 4 times of gate_size.
        input_forward_proj = fluid.layers.fc(input=input_seq,
                                             size=gate_size * 4,
                                             act='tanh',
                                             bias_attr=False)
        forward, _ = fluid.layers.dynamic_lstm(
            input=input_forward_proj, size=gate_size * 4, use_peepholes=False)
        input_reversed_proj = fluid.layers.fc(input=input_seq,
                                              size=gate_size * 4,
                                              act='tanh',
                                              bias_attr=False)
        reversed, _ = fluid.layers.dynamic_lstm(
            input=input_reversed_proj,
            size=gate_size * 4,
            is_reverse=True,
            use_peepholes=False)
        return forward, reversed

    # The encoding process. Encodes the input words into tensors.
    src_word_idx = fluid.layers.data(
        name='source_sequence', shape=[1], dtype='int64', lod_level=1)

    src_embedding = fluid.layers.embedding(
        input=src_word_idx,
        size=[source_dict_dim, embedding_dim],
        dtype='float32')

    src_forward, src_reversed = bi_lstm_encoder(
        input_seq=src_embedding, gate_size=encoder_size)

    encoded_vector = fluid.layers.concat(
        input=[src_forward, src_reversed], axis=1)

    encoded_proj = fluid.layers.fc(input=encoded_vector,
                                   size=decoder_size,
                                   bias_attr=False)

    backward_first = fluid.layers.sequence_pool(
        input=src_reversed, pool_type='first')

    decoder_boot = fluid.layers.fc(input=backward_first,
                                   size=decoder_size,
                                   bias_attr=False,
                                   act='tanh')

    cell_init = fluid.layers.fill_constant_batch_size_like(
        input=decoder_boot,
        value=0.0,
        shape=[-1, decoder_size],
        dtype='float32')
    cell_init.stop_gradient = False

    # Create a RNN state cell by providing the input and hidden states, and
    # specifies the hidden state as output.
    h = InitState(init=decoder_boot, need_reorder=True)
    c = InitState(init=cell_init)

    state_cell = StateCell(
        inputs={'x': None,
                'encoder_vec': None,
                'encoder_proj': None},
        states={'h': h,
                'c': c},
        out_state='h')

    def simple_attention(encoder_vec, encoder_proj, decoder_state):
        # The implementation of simple attention model
        decoder_state_proj = fluid.layers.fc(input=decoder_state,
                                             size=decoder_size,
                                             bias_attr=False)
        decoder_state_expand = fluid.layers.sequence_expand(
            x=decoder_state_proj, y=encoder_proj)
        # concated lod should inherit from encoder_proj
        mixed_state = encoder_proj + decoder_state_expand
        attention_weights = fluid.layers.fc(input=mixed_state,
                                            size=1,
                                            bias_attr=False)
        attention_weights = fluid.layers.sequence_softmax(
            input=attention_weights)
        weigths_reshape = fluid.layers.reshape(x=attention_weights, shape=[-1])
        scaled = fluid.layers.elementwise_mul(
            x=encoder_vec, y=weigths_reshape, axis=0)
        context = fluid.layers.sequence_pool(input=scaled, pool_type='sum')
        return context

    @state_cell.state_updater
    def state_updater(state_cell):
        # Define the updater of RNN state cell
        current_word = state_cell.get_input('x')
        encoder_vec = state_cell.get_input('encoder_vec')
        encoder_proj = state_cell.get_input('encoder_proj')
        prev_h = state_cell.get_state('h')
        prev_c = state_cell.get_state('c')
        context = simple_attention(encoder_vec, encoder_proj, prev_h)
        decoder_inputs = fluid.layers.concat(
            input=[context, current_word], axis=1)
        h, c = lstm_step(decoder_inputs, prev_h, prev_c, decoder_size)
        state_cell.set_state('h', h)
        state_cell.set_state('c', c)

    # Define the decoding process
    if not is_generating:
        # Training process
        trg_word_idx = fluid.layers.data(
            name='target_sequence', shape=[1], dtype='int64', lod_level=1)

        trg_embedding = fluid.layers.embedding(
            input=trg_word_idx,
            size=[target_dict_dim, embedding_dim],
            dtype='float32')

        # A decoder for training
        decoder = TrainingDecoder(state_cell)

        with decoder.block():
            current_word = decoder.step_input(trg_embedding)
            encoder_vec = decoder.static_input(encoded_vector)
            encoder_proj = decoder.static_input(encoded_proj)
            decoder.state_cell.compute_state(inputs={
                'x': current_word,
                'encoder_vec': encoder_vec,
                'encoder_proj': encoder_proj
            })
            h = decoder.state_cell.get_state('h')
            decoder.state_cell.update_states()
            out = fluid.layers.fc(input=h,
                                  size=target_dict_dim,
                                  bias_attr=True,
                                  act='softmax')
            decoder.output(out)

        label = fluid.layers.data(
            name='label_sequence', shape=[1], dtype='int64', lod_level=1)
        cost = fluid.layers.cross_entropy(input=decoder(), label=label)
        avg_cost = fluid.layers.mean(x=cost)
        feeding_list = ["source_sequence", "target_sequence", "label_sequence"]
        return avg_cost, feeding_list

    else:
        # Inference
        init_ids = fluid.layers.data(
            name="init_ids", shape=[1], dtype="int64", lod_level=2)
        init_scores = fluid.layers.data(
            name="init_scores", shape=[1], dtype="float32", lod_level=2)

        # A beam search decoder
        decoder = BeamSearchDecoder(
            state_cell=state_cell,
            init_ids=init_ids,
            init_scores=init_scores,
            target_dict_dim=target_dict_dim,
            word_dim=embedding_dim,
            input_var_dict={
                'encoder_vec': encoded_vector,
                'encoder_proj': encoded_proj
            },
            topk_size=50,
            sparse_emb=True,
            max_len=max_length,
            beam_size=beam_size,
            end_id=1,
            name=None)

        decoder.decode()

        translation_ids, translation_scores = decoder()
        feeding_list = ["source_sequence"]

        return translation_ids, translation_scores, feeding_list
