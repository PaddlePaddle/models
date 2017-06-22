#!/usr/bin/env python
# -*- coding: utf-8 -*-

import paddle.v2 as paddle
import sys
import math


def gru_encoder_decoder(src_dict_dim,
                        trg_dict_dim,
                        is_generating=False,
                        is_hybrid_addressing=True,
                        word_vec_dim=512,
                        latent_chain_dim=512,
                        beam_max_len=230,
                        beam_size=3):
    src_word_id = paddle.layer.data(
        name='source_language_word',
        type=paddle.data_type.integer_value_sequence(src_dict_dim))

    src_embedding = paddle.layer.embedding(
        input=src_word_id,
        size=word_vec_dim,
        param_attr=paddle.attr.ParamAttr(
            name='_source_language_embedding',
            initial_std=1. / math.sqrt(word_vec_dim)))
    # use bi-gru as encoder
    src_forward = paddle.networks.simple_gru(
        input=src_embedding, size=latent_chain_dim)
    src_backward = paddle.networks.simple_gru(
        input=src_embedding, size=latent_chain_dim, reverse=True)
    encoder_vector = paddle.layer.concat(input=[src_forward, src_backward])
    with paddle.layer.mixed(
            size=latent_chain_dim, bias_attr=False,
            act=paddle.activation.Linear()) as encoder_projected:
        encoder_projected += paddle.layer.full_matrix_projection(
            input=encoder_vector)

    if is_hybrid_addressing:
        attention_memory_init = paddle.layer.data(
            name='init_attention_weights',
            type=paddle.data_type.dense_vector(1))
        # expand dense vector to sequence
        expand_attention_memory_init = paddle.layer.expand(
            input=attention_memory_init, expand_as=src_word_id, bias_attr=False)

    # build decoder with/without addressing mechanism
    def gru_decoder_with_attention(encoder_projected, current_word):
        decoder_state_memory = paddle.layer.memory(
            name='gru_decoder', size=latent_chain_dim, is_seq=False)

        # get attention in this code section
        with paddle.layer.mixed(
                size=latent_chain_dim,
                act=paddle.activation.Linear(),
                bias_attr=False) as decoder_state_projected:
            decoder_state_projected += paddle.layer.full_matrix_projection(
                input=decoder_state_memory)
        expand_decoder_state_projected = paddle.layer.expand(
            input=decoder_state_projected,
            expand_as=encoder_projected,
            bias_attr=False)
        with paddle.layer.mixed(
                size=latent_chain_dim,
                act=paddle.activation.Tanh(),
                bias_attr=False) as attention_vecs:
            attention_vecs += paddle.layer.identity_projection(
                input=expand_decoder_state_projected)
            attention_vecs += paddle.layer.identity_projection(
                input=encoder_projected)
        with paddle.layer.mixed(
                name='attention_weights',
                size=1,
                act=paddle.activation.SequenceSoftmax(),
                bias_attr=False) as attention_weights:
            attention_weights += paddle.layer.full_matrix_projection(
                input=attention_vecs)

        if is_hybrid_addressing == False:
            context_vectors = paddle.layer.scaling(
                input=encoder_projected, weight=attention_weights)
        else:
            # save attention weights of last step
            attention_weight_memory = paddle.layer.memory(
                name='attention_weights',
                size=1,
                is_seq=True,
                boot_layer=expand_attention_memory_init)

            # interpolating weight
            with paddle.layer.mixed(
                    size=1, act=paddle.activation.Sigmoid(),
                    bias_attr=False) as addressing_gate:
                addressing_gate += paddle.layer.full_matrix_projection(
                    input=current_word)
            expand_addressing_gate = paddle.layer.expand(
                input=addressing_gate,
                expand_as=encoder_projected,
                bias_attr=False)
            weight_interpolation = paddle.layer.interpolation(
                input=[attention_weights, attention_weight_memory],
                weight=expand_addressing_gate)

            # convolution shift
            with paddle.layer.mixed(
                    size=3,
                    act=paddle.activation.Softmax(),
                    bias_attr=paddle.attr.Param(
                        initial_std=0)) as shifting_weights:
                shifting_weights += paddle.layer.full_matrix_projection(
                    input=current_word)
            convolutional_shift = paddle.layer.conv_shift(
                a=weight_interpolation, b=shifting_weights)
            context_vectors = paddle.layer.scaling(
                input=encoder_projected, weight=convolutional_shift)

        # sum together to get context vector  
        context = paddle.layer.pooling(
            input=context_vectors, pooling_type=paddle.pooling.Sum())

        with paddle.layer.mixed(
                size=latent_chain_dim * 3,
                layer_attr=paddle.attr.ExtraAttr(
                    error_clipping_threshold=100.0)) as decoder_step_input:
            decoder_step_input += paddle.layer.full_matrix_projection(
                input=context)
            decoder_step_input += paddle.layer.full_matrix_projection(
                input=current_word)

        gru_step = paddle.layer.gru_step(
            name='gru_decoder',
            input=decoder_step_input,
            output_mem=decoder_state_memory,
            size=latent_chain_dim)

        with paddle.layer.mixed(
                size=trg_dict_dim,
                act=paddle.activation.Softmax(),
                bias_attr=paddle.attr.Param(initial_std=0)) as out:
            out += paddle.layer.full_matrix_projection(input=gru_step)

        return out

    decoder_group_name = 'decoder_group'
    group_inputs = [
        paddle.layer.StaticInputV2(input=encoder_projected, is_seq=True)
    ]

    if not is_generating:
        trg_embedding = paddle.layer.embedding(
            input=paddle.layer.data(
                name='target_language_word',
                type=paddle.data_type.integer_value_sequence(trg_dict_dim)),
            size=word_vec_dim,
            param_attr=paddle.attr.ParamAttr(name='_target_language_embedding'))
        group_inputs.append(trg_embedding)
        decoder = paddle.layer.recurrent_group(
            name=decoder_group_name,
            step=gru_decoder_with_attention,
            input=group_inputs)
        lbl = paddle.layer.data(
            name='target_language_next_word',
            type=paddle.data_type.integer_value_sequence(trg_dict_dim))
        cost = paddle.layer.classification_cost(input=decoder, label=lbl)
        return cost
    else:
        trg_embedding = paddle.layer.GeneratedInputV2(
            size=trg_dict_dim,
            embedding_name='_target_language_embedding',
            embedding_size=word_vec_dim)
        group_inputs.append(trg_embedding)
        beam_gen = paddle.layer.beam_search(
            name=decoder_group_name,
            step=gru_decoder_with_attention,
            input=group_inputs,
            bos_id=0,
            eos_id=1,
            beam_size=beam_size,
            max_length=beam_max_len)
        return beam_gen
