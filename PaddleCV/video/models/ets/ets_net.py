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

import paddle.fluid as fluid
from paddle.fluid import ParamAttr
import numpy as np

DATATYPE = 'float32'


class ETSNET(object):
    def __init__(self,
                 feat_size,
                 fc_dim,
                 gru_hidden_dim,
                 max_length,
                 beam_size,
                 decoder_size,
                 word_emb_dim,
                 dict_size,
                 mode='train'):
        self.feat_size = feat_size
        self.fc_dim = fc_dim
        self.gru_hidden_dim = gru_hidden_dim
        self.decoder_size = decoder_size
        self.word_emb_dim = word_emb_dim
        self.dict_size = dict_size
        self.max_length = max_length
        self.beam_size = beam_size
        self.mode = mode

    def encoder(self, feat):
        bias_attr = fluid.ParamAttr(
            regularizer=fluid.regularizer.L2Decay(0.0),
            initializer=fluid.initializer.NormalInitializer(scale=0.0))

        input_fc = fluid.layers.fc(input=feat,
                                   size=self.fc_dim,
                                   act='tanh',
                                   bias_attr=bias_attr)
        gru_forward_fc = fluid.layers.fc(input=input_fc,
                                         size=self.gru_hidden_dim * 3,
                                         bias_attr=False)
        gru_forward = fluid.layers.dynamic_gru(
            input=gru_forward_fc, size=self.gru_hidden_dim, is_reverse=False)
        gru_backward_fc = fluid.layers.fc(input=input_fc,
                                          size=self.gru_hidden_dim * 3,
                                          bias_attr=False)
        gru_backward = fluid.layers.dynamic_gru(
            input=gru_backward_fc, size=self.gru_hidden_dim, is_reverse=True)
        encoded_sequence = fluid.layers.concat(
            input=[gru_forward, gru_backward], axis=1)
        gru_weights = fluid.layers.fc(input=encoded_sequence,
                                      size=1,
                                      act='sequence_softmax',
                                      bias_attr=False)
        gru_scaled = fluid.layers.elementwise_mul(
            x=encoded_sequence, y=gru_weights, axis=0)
        encoded_vector = fluid.layers.sequence_pool(
            input=gru_scaled, pool_type='sum')
        encoded_proj = fluid.layers.fc(input=encoded_sequence,
                                       size=self.decoder_size,
                                       bias_attr=False)
        return encoded_sequence, encoded_vector, encoded_proj

    def cell(self, x, hidden, encoder_out, encoder_out_proj):
        def simple_attention(encoder_vec, encoder_proj, decoder_state):
            decoder_state_proj = fluid.layers.fc(input=decoder_state,
                                                 size=self.decoder_size,
                                                 bias_attr=False)
            decoder_state_expand = fluid.layers.sequence_expand(
                x=decoder_state_proj, y=encoder_proj)
            mixed_state = fluid.layers.elementwise_add(encoder_proj,
                                                       decoder_state_expand)
            attention_weights = fluid.layers.fc(input=mixed_state,
                                                size=1,
                                                bias_attr=False)
            attention_weights = fluid.layers.sequence_softmax(
                input=attention_weights)
            weigths_reshape = fluid.layers.reshape(
                x=attention_weights, shape=[-1])
            scaled = fluid.layers.elementwise_mul(
                x=encoder_vec, y=weigths_reshape, axis=0)
            context = fluid.layers.sequence_pool(input=scaled, pool_type='sum')

            return context

        context = simple_attention(encoder_out, encoder_out_proj, hidden)
        out = fluid.layers.fc(input=[x, context],
                              size=self.decoder_size * 3,
                              bias_attr=False)
        out = fluid.layers.gru_unit(
            input=out, hidden=hidden, size=self.decoder_size * 3)[0]
        return out, out

    def train_decoder(self, word, encoded_sequence, encoded_vector,
                      encoded_proj):
        decoder_boot = fluid.layers.fc(input=encoded_vector,
                                       size=self.decoder_size,
                                       act='tanh',
                                       bias_attr=False)
        word_embedding = fluid.layers.embedding(
            input=word, size=[self.dict_size, self.word_emb_dim])

        pad_value = fluid.layers.assign(input=np.array([0.], dtype=np.float32))
        word_embedding, length = fluid.layers.sequence_pad(word_embedding,
                                                           pad_value)
        word_embedding = fluid.layers.transpose(word_embedding, [1, 0, 2])

        rnn = fluid.layers.StaticRNN()
        with rnn.step():
            x = rnn.step_input(word_embedding)
            pre_state = rnn.memory(init=decoder_boot)
            out, current_state = self.cell(x, pre_state, encoded_sequence,
                                           encoded_proj)
            prob = fluid.layers.fc(input=out,
                                   size=self.dict_size,
                                   act='softmax')

            rnn.update_memory(pre_state, current_state)
            rnn.step_output(prob)

        rnn_out = rnn()
        rnn_out = fluid.layers.transpose(rnn_out, [1, 0, 2])

        length = fluid.layers.reshape(length, [-1])
        rnn_out = fluid.layers.sequence_unpad(x=rnn_out, length=length)

        return rnn_out

    def infer_decoder(self, init_ids, init_scores, encoded_sequence,
                      encoded_vector, encoded_proj):
        decoder_boot = fluid.layers.fc(input=encoded_vector,
                                       size=self.decoder_size,
                                       act='tanh',
                                       bias_attr=False)

        max_len = fluid.layers.fill_constant(
            shape=[1], dtype='int64', value=self.max_length)
        counter = fluid.layers.zeros(shape=[1], dtype='int64', force_cpu=True)

        # create and init arrays to save selected ids, scores and states for each step
        ids_array = fluid.layers.array_write(init_ids, i=counter)
        scores_array = fluid.layers.array_write(init_scores, i=counter)
        state_array = fluid.layers.array_write(decoder_boot, i=counter)

        cond = fluid.layers.less_than(x=counter, y=max_len)
        while_op = fluid.layers.While(cond=cond)
        with while_op.block():
            pre_ids = fluid.layers.array_read(array=ids_array, i=counter)
            pre_score = fluid.layers.array_read(array=scores_array, i=counter)
            pre_state = fluid.layers.array_read(array=state_array, i=counter)

            pre_ids_emb = fluid.layers.embedding(
                input=pre_ids, size=[self.dict_size, self.word_emb_dim])

            out, current_state = self.cell(pre_ids_emb, pre_state,
                                           encoded_sequence, encoded_proj)
            prob = fluid.layers.fc(input=out,
                                   size=self.dict_size,
                                   act='softmax')

            # beam search
            topk_scores, topk_indices = fluid.layers.topk(
                prob, k=self.beam_size)
            accu_scores = fluid.layers.elementwise_add(
                x=fluid.layers.log(topk_scores),
                y=fluid.layers.reshape(
                    pre_score, shape=[-1]),
                axis=0)
            accu_scores = fluid.layers.lod_reset(x=accu_scores, y=pre_ids)
            selected_ids, selected_scores = fluid.layers.beam_search(
                pre_ids,
                pre_score,
                topk_indices,
                accu_scores,
                self.beam_size,
                end_id=1)

            fluid.layers.increment(x=counter, value=1, in_place=True)
            # save selected ids and corresponding scores of each step
            fluid.layers.array_write(selected_ids, array=ids_array, i=counter)
            fluid.layers.array_write(
                selected_scores, array=scores_array, i=counter)
            # update rnn state by sequence_expand acting as gather
            current_state = fluid.layers.sequence_expand(current_state,
                                                         selected_scores)
            fluid.layers.array_write(
                current_state, array=state_array, i=counter)
            current_enc_seq = fluid.layers.sequence_expand(encoded_sequence,
                                                           selected_scores)
            fluid.layers.assign(current_enc_seq, encoded_sequence)
            current_enc_proj = fluid.layers.sequence_expand(encoded_proj,
                                                            selected_scores)
            fluid.layers.assign(current_enc_proj, encoded_proj)

            # update conditional variable
            length_cond = fluid.layers.less_than(x=counter, y=max_len)
            finish_cond = fluid.layers.logical_not(
                fluid.layers.is_empty(x=selected_ids))
            fluid.layers.logical_and(x=length_cond, y=finish_cond, out=cond)

        translation_ids, translation_scores = fluid.layers.beam_search_decode(
            ids=ids_array,
            scores=scores_array,
            beam_size=self.beam_size,
            end_id=1)

        return translation_ids, translation_scores

    def net(self, feat, *input_decoder):
        encoded_sequence, encoded_vector, encoded_proj = self.encoder(feat)
        if (self.mode == 'train') or (self.mode == 'valid'):
            word, = input_decoder
            prob = self.train_decoder(word, encoded_sequence, encoded_vector,
                                      encoded_proj)
            return prob
        else:
            init_ids, init_scores = input_decoder
            translation_ids, translation_scores = self.infer_decoder(
                init_ids, init_scores, encoded_sequence, encoded_vector,
                encoded_proj)
            return translation_ids, translation_scores

    def loss(self, prob, word_next):
        cost = fluid.layers.cross_entropy(input=prob, label=word_next)
        avg_cost = fluid.layers.mean(cost)
        return [avg_cost]
