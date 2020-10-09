# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
import paddle
from paddle.nn import Layer, Linear, Dropout, Embedding, LayerList, RNN, LSTM, LSTMCell, RNNCellBase
import paddle.nn.initializer as I
import paddle.nn.functional as F
SEED = 102
paddle.framework.manual_seed(SEED)


class AttentionModel(Layer):
    def __init__(self,
                 hidden_size,
                 src_vocab_size,
                 trg_vocab_size,
                 num_layers=1,
                 init_scale=0.1,
                 pad_ids=0,
                 dropout=None,
                 beam_size=1,
                 beam_start_token=1,
                 beam_end_token=2,
                 beam_max_step_num=100):
        super(AttentionModel, self).__init__()
        self.hidden_size = hidden_size
        self.src_vocab_size = src_vocab_size
        self.trg_vocab_size = trg_vocab_size
        self.num_layers = num_layers
        self.init_scale = init_scale
        self.dropout = dropout
        self.beam_size = beam_size
        self.beam_start_token = beam_start_token
        self.beam_end_token = beam_end_token
        self.beam_max_step_num = beam_max_step_num
        self.kinf = 1e9

        self.encoder = Encoder(src_vocab_size, hidden_size, num_layers,
                               init_scale, pad_ids[0], dropout)
        self.decoder = Decoder(trg_vocab_size, hidden_size, num_layers,
                               init_scale, pad_ids[1], dropout)

    def forward(self, inputs):

        src, trg, label, src_seq_len, trg_seq_len = inputs
        enc_states, enc_outputs, enc_padding_mask = self.encoder(src,
                                                                 src_seq_len)
        enc_states = [(enc_states[0][i], enc_states[1][i])
                      for i in range(self.num_layers)]
        dec_output, trg_mask = self.decoder(trg, trg_seq_len, enc_states,
                                            enc_outputs, enc_padding_mask)
        loss = self.calc_loss(dec_output, trg_mask, label)

        return loss

    def calc_loss(self, dec_output, trg_mask, label):
        loss = F.softmax_with_cross_entropy(
            logits=dec_output, label=label, soft_label=False)
        loss = paddle.squeeze(loss, axis=[2])
        loss = loss * trg_mask
        loss = paddle.reduce_mean(loss, dim=[0])
        loss = paddle.reduce_sum(loss)
        return loss


class Encoder(Layer):
    def __init__(self,
                 vocab_size,
                 hidden_size,
                 num_layers=1,
                 init_scale=0.1,
                 padding_idx=0,
                 dropout=None):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.padding_idx = padding_idx
        self.embedder = Embedding(
            vocab_size,
            hidden_size,
            padding_idx=padding_idx,
            weight_attr=paddle.ParamAttr(initializer=I.Uniform(
                low=-init_scale, high=init_scale)))
        self.lstm = LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            direction="forward",
            dropout=dropout if num_layers > 1 else 0., )

    def forward(self, src, src_sequence_length):
        src_emb = self.embedder(src)

        outs, (final_h, final_c) = self.lstm(
            src_emb, sequence_length=src_sequence_length)

        enc_len_mask = (
            src != self.padding_idx).astype(paddle.get_default_dtype())
        enc_padding_mask = (enc_len_mask - 1.0) * 1e9
        return [final_h, final_c], outs, enc_padding_mask


class AttentionLayer(Layer):
    def __init__(self, hidden_size, bias=False, init_scale=0.1):
        super(AttentionLayer, self).__init__()
        self.input_proj = Linear(
            hidden_size,
            hidden_size,
            weight_attr=paddle.ParamAttr(initializer=I.Uniform(
                low=-init_scale, high=init_scale)),
            bias_attr=bias)
        self.output_proj = Linear(
            hidden_size + hidden_size,
            hidden_size,
            weight_attr=paddle.ParamAttr(initializer=I.Uniform(
                low=-init_scale, high=init_scale)),
            bias_attr=bias)

    def forward(self, hidden, encoder_output, encoder_padding_mask):
        query = self.input_proj(hidden)

        attn_scores = paddle.matmul(
            paddle.unsqueeze(query, [1]), encoder_output, transpose_y=True)
        encoder_padding_mask = paddle.unsqueeze(encoder_padding_mask, [1])
        if encoder_padding_mask is not None:
            attn_scores = paddle.add(attn_scores, encoder_padding_mask)
        attn_scores = F.softmax(attn_scores)

        attn_out = paddle.matmul(attn_scores, encoder_output)
        attn_out = paddle.squeeze(attn_out, [1])
        attn_out = paddle.concat([attn_out, hidden], 1)
        attn_out = self.output_proj(attn_out)
        return attn_out


class DecoderCell(RNNCellBase):
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers,
                 init_scale=0.1,
                 dropout=0.):
        super(DecoderCell, self).__init__()
        if dropout > 0.0:
            self.dropout = Dropout(dropout)
        else:
            self.dropout = None
        self.lstm_cells = LayerList([
            LSTMCell(
                input_size=input_size + hidden_size if i == 0 else hidden_size,
                hidden_size=hidden_size) for i in range(num_layers)
        ])
        self.attention_layer = AttentionLayer(hidden_size)

    def forward(self,
                step_input,
                states,
                encoder_output,
                encoder_padding_mask=None):
        lstm_states, input_feed = states
        new_lstm_states = []

        step_input = paddle.concat([step_input, input_feed], 1)
        for i, lstm_cell in enumerate(self.lstm_cells):

            new_hidden, (new_hidden, new_cell) = lstm_cell(step_input,
                                                           lstm_states[i])
            if self.dropout:
                new_hidden = self.dropout(new_hidden)

            new_lstm_state = [new_hidden, new_cell]
            new_lstm_states.append(new_lstm_state)
            step_input = new_hidden
        out = self.attention_layer(step_input, encoder_output,
                                   encoder_padding_mask)
        return out, [new_lstm_states, out]


class Decoder(Layer):
    def __init__(self,
                 vocab_size,
                 hidden_size,
                 num_layers=1,
                 init_scale=0.1,
                 padding_idx=0,
                 dropout=None):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.init_scale = init_scale
        self.padding_idx = padding_idx
        self.embedder = Embedding(
            vocab_size,
            hidden_size,
            padding_idx=padding_idx,
            weight_attr=paddle.ParamAttr(initializer=I.Uniform(
                low=-init_scale, high=init_scale)))
        self.dropout = dropout
        self.lstm_attention = RNN(DecoderCell(hidden_size, hidden_size,
                                              num_layers, init_scale, dropout),
                                  is_reverse=False,
                                  time_major=False)
        self.fc = Linear(
            hidden_size,
            vocab_size,
            weight_attr=paddle.ParamAttr(initializer=I.Uniform(
                low=-init_scale, high=init_scale)),
            bias_attr=False)

    def forward(self, trg, trg_sequence_length, enc_states, enc_outputs,
                enc_padding_mask):
        trg_emb = self.embedder(trg)
        bsz = paddle.shape(trg)[0]
        input_feed = paddle.zeros(
            (bsz, self.hidden_size), dtype=paddle.get_default_dtype())
        states = [enc_states, input_feed]
        dec_output, _ = self.lstm_attention(
            trg_emb,
            initial_states=states,
            encoder_output=enc_outputs,
            encoder_padding_mask=enc_padding_mask)

        dec_output = self.fc(dec_output)
        trg_mask = (trg != self.padding_idx).astype(paddle.get_default_dtype())
        return dec_output, trg_mask
