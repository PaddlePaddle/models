#!/usr/bin/env python
# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
File: source/models/knowledge_seq2seq.py
"""

import numpy as np
import paddle.fluid as fluid
import paddle.fluid.layers as layers
from paddle.fluid.layers.control_flow import StaticRNN as PaddingRNN
from source.utils.utils import log_softmax


def get_embedding(input, emb_size, vocab_size, name=""):
    """ get embedding """
    return layers.embedding(input,
                            size=[vocab_size, emb_size],
                            param_attr=fluid.ParamAttr(name="embedding"),
                            is_sparse=True)


def fc(input, input_size, output_size, bias=True, name="fc"):
    """ fc """
    weight = layers.create_parameter([input_size, output_size],
                                     dtype='float32',
                                     name=name + "_w")
    out = layers.matmul(input, weight)
    if bias:
        bias = layers.create_parameter([output_size],
                                       dtype='float32',
                                       name=name + "_b")
        out = out + bias

    return out


class GRU_unit(object):
    """ GRU unit """
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.0, name="gru_unit"):
        self.weight_input_array = []
        self.weight_hidden_array = []
        self.bias_input_array = []
        self.bias_hidden_array = []
        self.init_hidden_array = []

        # init gru param
        for i in range(num_layers):
            weight_input = layers.create_parameter([input_size, hidden_size * 3],
                                                   dtype='float32',
                                                   name=name + "_input_w")
            self.weight_input_array.append(weight_input)
            weight_hidden = layers.create_parameter([hidden_size, hidden_size * 3],
                                                    dtype='float32',
                                                    name=name + "_hidden_w")
            self.weight_hidden_array.append(weight_hidden)
            bias_input = layers.create_parameter([hidden_size * 3],
                                                 dtype='float32',
                                                 name=name + "_input_b")
            self.bias_input_array.append(bias_input)
            bias_hidden = layers.create_parameter([hidden_size * 3],
                                                  dtype='float32',
                                                  name=name + "_hidden_b")
            self.bias_hidden_array.append(bias_hidden)

        self.dropout = dropout
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size

    def gru_step(self, input, hidden, mask=None):
        """ gru step """
        hidden_array = []
        for i in range(self.num_layers):
            hidden_temp = layers.slice(hidden, axes = [0], starts = [i], ends = [i + 1])
            hidden_temp = layers.reshape(hidden_temp, shape=[-1, self.hidden_size])
            hidden_array.append(hidden_temp)

        last_hidden_array = []
        for k in range(self.num_layers):
            trans_input = layers.matmul(input, self.weight_input_array[k])
            trans_input += self.bias_input_array[k]
            trans_hidden = layers.matmul(hidden_array[k], self.weight_hidden_array[k])
            trans_hidden += self.bias_hidden_array[k]

            input_array = layers.split(trans_input, num_or_sections=3, dim=-1)
            trans_array = layers.split(trans_hidden, num_or_sections=3, dim=-1)

            reset_gate = layers.sigmoid(input_array[0] + trans_array[0])
            input_gate = layers.sigmoid(input_array[1] + trans_array[1])
            new_gate = layers.tanh(input_array[2] + reset_gate * trans_array[2])
            
            new_hidden = new_gate + input_gate * (hidden_array[k] - new_gate)

            if mask:
                neg_mask = layers.fill_constant_batch_size_like(input=mask,
                                                                shape=[1],
                                                                value=1.0,
                                                                dtype='float32') - mask
                new_hidden = new_hidden * mask + hidden_array[k] * neg_mask

            last_hidden_array.append(new_hidden)
            input = new_hidden

            if self.dropout and self.dropout > 0.0:
                input = layers.dropout(input, dropout_prob = self.dropout)

        last_hidden = layers.concat(last_hidden_array, 0)
        last_hidden = layers.reshape(last_hidden,
                                     shape=[self.num_layers, -1, self.hidden_size])

        return input, last_hidden

    def __call__(self, input, hidden, mask=None):
        return self.gru_step(input, hidden, mask)


def gru_rnn(input, input_size, hidden_size,
            init_hidden=None, batch_first=False,
            mask=None, num_layers=1, dropout=0.0, name="gru"):
    """ gru rnn """

    gru_unit = GRU_unit(input_size, hidden_size,
                        num_layers=num_layers, dropout=dropout, name=name + "_gru_unit")
    
    if batch_first:
        input = layers.transpose(x=input, perm=[1, 0, 2])
        if mask:
            mask = layers.transpose(mask, perm=[1, 0])

    rnn = PaddingRNN()
    with rnn.step():
        step_in = rnn.step_input(input)
        step_mask = None

        if mask:
            step_mask = rnn.step_input(mask)

        pre_hidden = rnn.memory(init = init_hidden)
        new_hidden, last_hidden = gru_unit(step_in, pre_hidden, step_mask)
        rnn.update_memory(pre_hidden, last_hidden)
        step_in = new_hidden
        rnn.step_output(step_in)
        rnn.step_output(last_hidden)

    rnn_res = rnn()
    rnn_out = rnn_res[0]
    last_hidden = layers.slice(rnn_res[1], axes=[0], starts=[-1], ends=[1000000000])
    last_hidden = layers.reshape(last_hidden, shape=[num_layers, -1, hidden_size])

    if batch_first:
        rnnout = layers.transpose(x = rnn_out, perm=[1, 0, 2])

    return rnnout, last_hidden


def bidirec_gru(input, input_size, hidden_size,
                batch_size, batch_first=True, num_layers=1,
                dropout=0.0, mask=None, last_mask=None, name='bidir_gru'):
    """ bidirec gru """

    # use lod dynamic gru
    def gru_fun(gru_in, name=None, is_reverse=False):
        """ gru fun """
        fw_last_array = []
        fw_in = gru_in
        for i in range(num_layers):
            fw_gru_in = layers.fc(input=fw_in, size=hidden_size * 3,
                                  param_attr=fluid.ParamAttr(name=name + "_fc_w"),
                                  bias_attr=fluid.ParamAttr(name=name + "_fc_b"))
            fw_gru_out = layers.dynamic_gru(input=fw_gru_in, size=hidden_size,
                                            param_attr= fluid.ParamAttr(name=name + "_w"),
                                            bias_attr=fluid.ParamAttr(name=name + "_b"),
                                            origin_mode=True, is_reverse=is_reverse)
            fw_in = fw_gru_out

            if is_reverse:
                fw_last_hidden = layers.sequence_first_step(fw_gru_out)
            else:
                fw_last_hidden = layers.sequence_last_step(fw_gru_out)
            
            if last_mask:
                fw_last_hidden = layers.elementwise_mul(fw_last_hidden, last_mask, axis=0)

            fw_last_array.append(fw_last_hidden)

        if num_layers == 1:
            final_fw_last_hidden = layers.unsqueeze(fw_last_array[0], axes=[0])
        else:
            final_fw_last_hidden = layers.concat(fw_last_array, axis=0)
            final_fw_last_hidden = layers.reshape(final_fw_last_hidden,
                                                  shape=[num_layers, -1, hidden_size])
        
        final_fw_out = fw_in
        return final_fw_out, final_fw_last_hidden
    
    fw_rnn_out, fw_last_hidden = gru_fun(input, name=name + "_fw")
    bw_rnn_out, bw_last_hidden = gru_fun(input, name=name + "_bw", is_reverse=True)

    return [fw_rnn_out, bw_rnn_out, fw_last_hidden, bw_last_hidden]


def dot_attention(query, memory, mask=None):
    """ dot attention """
    attn = layers.matmul(query, memory, transpose_y=True)

    if mask:
        attn += mask * -1000000000

    weight = layers.softmax(attn)
    weight_memory = layers.matmul(weight, memory)

    return weight_memory, weight


def rnn_encoder(input, vocab_size, input_size, hidden_size,
                batch_size, num_layers, bi_direc, dropout=0.0,
                batch_first=True, mask=None, last_mask=None, name="rnn_enc"):
    """ rnn encoder """
    input_emb = get_embedding(input, input_size, vocab_size)
    fw_rnn_out, bw_rnn_out, fw_last_hidden, bw_last_hidden = \
        bidirec_gru(input_emb, input_size, hidden_size, batch_size,
                    batch_first=batch_first, num_layers = num_layers,
                    dropout=dropout, mask=mask, last_mask = last_mask, name=name)

    output = layers.concat([fw_rnn_out, bw_rnn_out], axis = 1)
    last_hidden = layers.concat([fw_last_hidden, bw_last_hidden], axis= 2)
        
    return output, last_hidden


def decoder_step(gru_unit, cue_gru_unit, step_in,
                 hidden, input_size, hidden_size,
                 memory, memory_mask, knowledge, mask=None):
    """ decoder step """
    # get attention out
    # get hidden top layers
    top_hidden = layers.slice(hidden, axes=[0], starts=[0], ends=[1])
    top_hidden = layers.squeeze(top_hidden, axes=[0])
    top_hidden = layers.unsqueeze(top_hidden, axes=[1])

    weight_memory, attn = dot_attention(top_hidden, memory, memory_mask)

    step_in = layers.unsqueeze(step_in, axes=[1])
    rnn_input_list = [step_in, weight_memory]
    if weight_memory.shape[0] == -1:
        knowledge_1 = layers.reshape(knowledge, shape=weight_memory.shape)
    else:
        knowledge_1 = knowledge
    cue_input_list = [knowledge_1, weight_memory]
    output_list = [weight_memory]

    rnn_input = layers.concat(rnn_input_list, axis=2)

    rnn_input = layers.squeeze(rnn_input, axes=[1])
    rnn_output, rnn_last_hidden = gru_unit(rnn_input, hidden, mask)

    cue_input = layers.concat(cue_input_list, axis=2)
    cue_input = layers.squeeze(cue_input, axes=[1])
    cue_rnn_out, cue_rnn_last_hidden = cue_gru_unit(cue_input, hidden, mask)

    h_y = layers.tanh(fc(rnn_last_hidden, hidden_size, hidden_size, name="dec_fc1"))
    h_cue = layers.tanh(fc(cue_rnn_last_hidden, hidden_size, hidden_size, name="dec_fc2"))

    concate_y_cue = layers.concat([h_y, h_cue], axis=2)
    k = layers.sigmoid(fc(concate_y_cue, hidden_size * 2, 1, name='dec_fc3'))

    new_hidden = h_y * k - h_cue * (k - 1.0)

    new_hidden_tmp = layers.transpose(new_hidden, perm=[1, 0, 2])
    output_list.append(new_hidden_tmp)

    real_out = layers.concat(output_list, axis=2)

    if mask:
        mask_tmp = layers.unsqueeze(mask, axes=[0])
        new_hidden = layers.elementwise_mul((new_hidden - hidden), mask_tmp, axis=0)
        new_hidden += hidden

    return real_out, new_hidden


def rnn_decoder(gru_unit, cue_gru_unit, input, input_size, hidden_size,
                num_layers, memory, memory_mask, knowledge, output_size,
                init_hidden=None, mask=None, dropout=0.0, batch_first=True, name="decoder"):
    """ rnn decoder """
    input_emb = get_embedding(input, input_size, output_size)
    if batch_first:
        input_emb = layers.transpose(input_emb, perm=[1, 0, 2])
        if mask:
            trans_mask = layers.transpose(mask, perm=[1, 0])

    rnn = PaddingRNN()
    with rnn.step():
        step_in = rnn.step_input(input_emb)
        step_mask = None

        if mask:
            step_mask = rnn.step_input(trans_mask)

        # split pre_hidden
        pre_hidden_list = []

        pre_hidden = rnn.memory(init = init_hidden)
        real_out, last_hidden = \
            decoder_step(gru_unit, cue_gru_unit, step_in, pre_hidden, input_size,
                         hidden_size, memory, memory_mask, knowledge, mask=step_mask)

        rnn.update_memory(pre_hidden, last_hidden)

        step_in = layers.squeeze(real_out, axes=[1])
        rnn.step_output(step_in)

    rnnout = rnn()
    rnnout = layers.transpose(rnnout, perm=[1, 0, 2])
    rnnout = layers.elementwise_mul(rnnout, mask, axis=0)

    output_in_size = hidden_size + hidden_size
    rnnout = layers.dropout(rnnout, dropout_prob = dropout)
    rnnout = fc(rnnout, output_in_size, hidden_size, name='dec_out_fc1')
    rnnout = fc(rnnout, hidden_size, output_size, name='dec_out_fc2')

    softmax_out = layers.softmax(rnnout)

    return softmax_out


def knowledge_seq2seq(config):
    """ knowledge seq2seq """
    emb_size = config.embed_size
    hidden_size = config.hidden_size
    input_size = emb_size
    num_layers = config.num_layers
    bi_direc = config.bidirectional
    batch_size = config.batch_size
    vocab_size = config.vocab_size
    run_type = config.run_type

    enc_input = layers.data(name="enc_input", shape=[1], dtype='int64', lod_level=1)
    enc_mask = layers.data(name="enc_mask", shape=[-1, 1], dtype='float32')
    cue_input = layers.data(name="cue_input", shape=[1], dtype='int64', lod_level=1)
    #cue_mask = layers.data(name='cue_mask', shape=[-1, 1], dtype='float32')
    memory_mask = layers.data(name='memory_mask', shape=[-1, 1], dtype='float32')
    tar_input = layers.data(name='tar_input', shape=[1], dtype='int64', lod_level=1)
    # tar_mask = layers.data(name="tar_mask", shape=[-1, 1], dtype='float32')

    rnn_hidden_size = hidden_size
    if bi_direc:
        rnn_hidden_size //= 2

    enc_out, enc_last_hidden = \
        rnn_encoder(enc_input, vocab_size, input_size, rnn_hidden_size, batch_size, num_layers, bi_direc,
                    dropout=0.0, batch_first=True, name="rnn_enc")

    bridge_out = fc(enc_last_hidden, hidden_size, hidden_size, name="bridge")
    bridge_out = layers.tanh(bridge_out)

    cue_last_mask = layers.data(name='cue_last_mask', shape=[-1], dtype='float32')
    knowledge_out, knowledge_last_hidden = \
        rnn_encoder(cue_input, vocab_size, input_size, rnn_hidden_size, batch_size, num_layers, bi_direc,
                    dropout=0.0, batch_first=True, last_mask=cue_last_mask, name="knowledge_enc")

    query = layers.slice(bridge_out, axes=[0], starts=[0], ends=[1])
    query = layers.squeeze(query, axes=[0])
    query = layers.unsqueeze(query, axes=[1])
    query = layers.reshape(query, shape=[batch_size, -1, hidden_size])
    cue_memory = layers.slice(knowledge_last_hidden, axes=[0], starts=[0], ends=[1])
    cue_memory = layers.reshape(cue_memory, shape=[batch_size, -1, hidden_size])
    memory_mask = layers.reshape(memory_mask, shape=[batch_size, 1, -1])

    weighted_cue, cue_att = dot_attention(query, cue_memory, mask=memory_mask)

    cue_att = layers.reshape(cue_att, shape=[batch_size, -1])

    knowledge = weighted_cue
    if config.use_posterior:
        target_out, target_last_hidden = \
            rnn_encoder(tar_input, vocab_size, input_size, rnn_hidden_size, batch_size, num_layers, bi_direc,
                        dropout=0.0, batch_first=True, name="knowledge_enc")

        # get attenion
        target_query = layers.slice(target_last_hidden, axes=[0], starts=[0], ends=[1])
        target_query = layers.squeeze(target_query, axes=[0])
        target_query = layers.unsqueeze(target_query, axes=[1])
        target_query = layers.reshape(target_query, shape=[batch_size, -1, hidden_size])

        weight_target, target_att = dot_attention(target_query, cue_memory, mask=memory_mask)
        target_att = layers.reshape(target_att, shape=[batch_size, -1])
        # add to output
        knowledge = weight_target

    enc_memory_mask = layers.data(name="enc_memory_mask", shape=[-1, 1], dtype='float32')
    enc_memory_mask = layers.unsqueeze(enc_memory_mask, axes=[1])
    # decoder init_hidden, enc_memory, enc_mask
    dec_init_hidden = bridge_out
    pad_value = fluid.layers.assign(
             input=np.array([0.0], dtype='float32'))
    
    enc_memory, origl_len_1 = layers.sequence_pad(x = enc_out, pad_value=pad_value)
    enc_memory.persistable = True

    gru_unit = GRU_unit(input_size + hidden_size, hidden_size,
                        num_layers=num_layers, dropout=0.0, name="decoder_gru_unit")

    cue_gru_unit = GRU_unit(hidden_size + hidden_size, hidden_size,
                            num_layers=num_layers, dropout=0.0, name="decoder_cue_gru_unit")

    tgt_vocab_size = config.vocab_size
    if run_type == "train":
        if config.use_bow:
            bow_logits = fc(knowledge, hidden_size, hidden_size, name='bow_fc_1')
            bow_logits = layers.tanh(bow_logits)
            bow_logits = fc(bow_logits, hidden_size, tgt_vocab_size, name='bow_fc_2')
            bow_logits = layers.softmax(bow_logits)

            bow_label = layers.data(name='bow_label', shape=[-1, config.max_len], dtype='int64')
            bow_mask = layers.data(name="bow_mask", shape=[-1, config.max_len], dtype='float32')

            bow_logits = layers.expand(bow_logits, [1, config.max_len, 1])
            bow_logits = layers.reshape(bow_logits, shape=[-1, tgt_vocab_size])
            bow_label = layers.reshape(bow_label, shape=[-1, 1])
            bow_loss = layers.cross_entropy(bow_logits, bow_label, soft_label=False)
            bow_loss = layers.reshape(bow_loss, shape=[-1, config.max_len])

            bow_loss *= bow_mask
            bow_loss = layers.reduce_sum(bow_loss, dim=[1])
            bow_loss = layers.reduce_mean(bow_loss)

        dec_input = layers.data(name="dec_input", shape=[-1, 1, 1], dtype='int64')
        dec_mask = layers.data(name="dec_mask", shape=[-1, 1], dtype='float32')

        dec_knowledge = weight_target

        decoder_logits = \
            rnn_decoder(gru_unit, cue_gru_unit, dec_input, input_size, hidden_size, num_layers,
                         enc_memory, enc_memory_mask, dec_knowledge, vocab_size,
                         init_hidden=dec_init_hidden, mask=dec_mask, dropout=config.dropout)

        target_label = layers.data(name='target_label', shape=[-1, 1], dtype='int64')
        target_mask = layers.data(name='target_mask', shape=[-1, 1], dtype='float32')

        decoder_logits = layers.reshape(decoder_logits, shape=[-1, tgt_vocab_size])
        target_label = layers.reshape(target_label, shape=[-1, 1])

        nll_loss = layers.cross_entropy(decoder_logits, target_label, soft_label = False)
        nll_loss = layers.reshape(nll_loss, shape=[batch_size, -1])
        nll_loss *= target_mask
        nll_loss = layers.reduce_sum(nll_loss, dim=[1])
        nll_loss = layers.reduce_mean(nll_loss)

        prior_attn = cue_att + 1e-10
        posterior_att = target_att
        posterior_att.stop_gradient = True

        prior_attn = layers.log(prior_attn)

        kl_loss = posterior_att * (layers.log(posterior_att + 1e-10) - prior_attn)
        kl_loss = layers.reduce_mean(kl_loss)

        kl_and_nll_factor = layers.data(name='kl_and_nll_factor', shape=[1], dtype='float32')
        kl_and_nll_factor = layers.reshape(kl_and_nll_factor, shape=[-1])


        final_loss = bow_loss + kl_loss * kl_and_nll_factor + nll_loss * kl_and_nll_factor

        return [bow_loss, kl_loss, nll_loss, final_loss]

    elif run_type == "test":
        beam_size = config.beam_size
        batch_size = config.batch_size
        token = layers.fill_constant(shape=[batch_size * beam_size, 1], 
                                     value=config.bos_id, dtype='int64')

        token = layers.reshape(token, shape=[-1, 1])
        max_decode_len = config.max_dec_len

        dec_knowledge = knowledge
        INF= 100000000.0

        init_score_np = np.ones([beam_size * batch_size], dtype='float32') * -INF

        for i in range(batch_size):
            init_score_np[i * beam_size] = 0.0

        pre_score = layers.assign(init_score_np)

        pos_index_np = np.arange(batch_size).reshape(-1, 1)
        pos_index_np = \
            np.tile(pos_index_np, (1, beam_size)).reshape(-1).astype('int32') * beam_size

        pos_index = layers.assign(pos_index_np)

        id_array = []
        score_array = []
        index_array = []
        init_enc_memory = layers.expand(enc_memory, [1, beam_size, 1])
        init_enc_memory = layers.reshape(init_enc_memory,
                                         shape=[batch_size * beam_size, -1, hidden_size])
        init_enc_mask = layers.expand(enc_memory_mask, [1, beam_size, 1])
        init_enc_mask = layers.reshape(init_enc_mask, shape=[batch_size * beam_size, 1, -1])

        dec_knowledge = layers.reshape(dec_knowledge, shape=[-1, 1, hidden_size])
        init_dec_knowledge = layers.expand(dec_knowledge, [1, beam_size, 1])
        init_dec_knowledge = layers.reshape(init_dec_knowledge,
                                            shape=[batch_size * beam_size, -1, hidden_size])

        dec_init_hidden = layers.expand(dec_init_hidden, [1, 1, beam_size])
        dec_init_hidden = layers.reshape(dec_init_hidden, shape=[1, -1, hidden_size])

        length_average = config.length_average
        UNK = config.unk_id
        EOS = config.eos_id
        for i in range(1, max_decode_len + 1):
            dec_emb = get_embedding(token, input_size, vocab_size)
            dec_out, dec_last_hidden = \
                decoder_step(gru_unit, cue_gru_unit,
                             dec_emb, dec_init_hidden, input_size, hidden_size,
                             init_enc_memory, init_enc_mask, init_dec_knowledge, mask=None)
            output_in_size = hidden_size + hidden_size

            rnnout = layers.dropout(dec_out, dropout_prob=config.dropout, is_test = True)
            rnnout = fc(rnnout, output_in_size, hidden_size, name='dec_out_fc1')
            rnnout = fc(rnnout, hidden_size, vocab_size, name='dec_out_fc2')

            log_softmax_output = log_softmax(rnnout)
            log_softmax_output = layers.squeeze(log_softmax_output, axes=[1])
            
            if i > 1:
                if length_average:
                    log_softmax_output = layers.elementwise_add((log_softmax_output / i),
                                                                (pre_score * (1.0 - 1.0 / i)),
                                                                axis=0)
                else:
                    log_softmax_output = layers.elementwise_add(log_softmax_output,
                                                                pre_score, axis=0)
            else:
                log_softmax_output = layers.elementwise_add(log_softmax_output,
                                                            pre_score, axis=0)

            log_softmax_output = layers.reshape(log_softmax_output, shape=[batch_size, -1])

            topk_score, topk_index = layers.topk(log_softmax_output, k = beam_size)
            topk_score = layers.reshape(topk_score, shape=[-1])
            topk_index = layers.reshape(topk_index, shape =[-1])
            
            vocab_var = layers.fill_constant([1], dtype='int64', value=vocab_size) 
            new_token = topk_index % vocab_var

            index = topk_index // vocab_var
            id_array.append(new_token)
            index_array.append(index)
            index = index + pos_index

            score_array.append(topk_score)

            eos_ids = layers.fill_constant([beam_size * batch_size], dtype='int64', value=EOS)
            unk_ids = layers.fill_constant([beam_size * batch_size], dtype='int64', value=UNK)
            eos_eq = layers.cast(layers.equal(new_token, eos_ids), dtype='float32')

            topk_score += eos_eq * -100000000.0

            unk_eq = layers.cast(layers.equal(new_token, unk_ids), dtype='float32')
            topk_score += unk_eq * -100000000.0

            # update
            token = new_token
            pre_score = topk_score
            token = layers.reshape(token, shape=[-1, 1])

            index = layers.cast(index, dtype='int32')
            dec_last_hidden = layers.squeeze(dec_last_hidden, axes=[0])
            dec_init_hidden = layers.gather(dec_last_hidden, index=index)
            dec_init_hidden = layers.unsqueeze(dec_init_hidden, axes=[0])
            init_enc_memory = layers.gather(init_enc_memory, index)
            init_enc_mask = layers.gather(init_enc_mask, index)
            init_dec_knowledge = layers.gather(init_dec_knowledge, index)

        final_score = layers.concat(score_array, axis=0)
        final_ids = layers.concat(id_array, axis=0)
        final_index = layers.concat(index_array, axis = 0)

        final_score = layers.reshape(final_score, shape=[max_decode_len, beam_size * batch_size])
        final_ids = layers.reshape(final_ids, shape=[max_decode_len, beam_size * batch_size])
        final_index = layers.reshape(final_index, shape=[max_decode_len, beam_size * batch_size])

        return final_score, final_ids, final_index
