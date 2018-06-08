# -*- coding:utf-8 -*-
import sys, os
import numpy as np
import paddle.v2 as paddle

def two_layer_lstm_net(source_dict_dim,
                   target_dict_dim,
                   is_generating,
                   beam_size=3,
                   max_length=250):
    ### 网络结构定义
    word_vector_dim = 512  # 词向量维度
    decoder_size = 512  # decoder隐藏单元的维度
    encoder_size = 512  # encoder隐藏单元维度

    #### Encoder 输入疑问句
    src_word_id = paddle.layer.data(
        name='source_words',
        type=paddle.data_type.integer_value_sequence(source_dict_dim))
    src_embedding = paddle.layer.embedding(
        input=src_word_id, size=word_vector_dim)
    src_forward = paddle.networks.simple_gru(
        input=src_embedding, size=encoder_size)
    src_backward = paddle.networks.simple_gru(
        input=src_embedding, size=encoder_size, reverse=True)
    encoded_vector = paddle.layer.concat(input=[src_forward, src_backward])

    #### Decoder 输出回答
    encoded_proj = paddle.layer.fc(
        act=paddle.activation.Linear(),
        size=decoder_size,
        bias_attr=False,
        input=encoded_vector)

    backward_first = paddle.layer.first_seq(input=src_backward)

    decoder_boot = paddle.layer.fc(
        size=decoder_size,
        act=paddle.activation.Tanh(),
        bias_attr=False,
        input=backward_first)

    def two_layer_lstm_decoder(enc_vec, enc_proj, current_word):

        decoder_mem = paddle.layer.memory(
            name='two_layer_lstm_decoder', size=decoder_size, boot_layer=decoder_boot)

        context = paddle.networks.simple_attention(
            encoded_sequence=enc_vec,
            encoded_proj=enc_proj,
            decoder_state=decoder_mem)

        decoder_inputs = paddle.layer.fc(
            act=paddle.activation.Linear(),
            size=decoder_size * 3,
            bias_attr=False,
            input=[context, current_word],
            layer_attr=paddle.attr.ExtraLayerAttribute(
                error_clipping_threshold=100.0))

        two_layer_lstm_step = paddle.layer.gru_step(
            name='two_layer_lstm_decoder',
            input=decoder_inputs,
            output_mem=decoder_mem,
            size=decoder_size)

        out = paddle.layer.fc(
            size=target_dict_dim,
            bias_attr=True,
            act=paddle.activation.Softmax(),
            input=two_layer_lstm_step)
        return out

    decoder_group_name = 'decoder_group'
    group_input1 = paddle.layer.StaticInput(input=encoded_vector)
    group_input2 = paddle.layer.StaticInput(input=encoded_proj)
    group_inputs = [group_input1, group_input2]

    if not is_generating:
        trg_embedding = paddle.layer.embedding(
            input=paddle.layer.data(
                name='two_layer_lstm_reader',
                type=paddle.data_type.integer_value_sequence(target_dict_dim)),
            size=word_vector_dim,
            param_attr=paddle.attr.ParamAttr(name='_two_layer_lstm_reader_embedding'))
        group_inputs.append(trg_embedding)

        decoder = paddle.layer.recurrent_group(
            name=decoder_group_name,
            step=two_layer_lstm_decoder,
            input=group_inputs)

        lbl = paddle.layer.data(
            name='two_layer_lstm_reader_next_word',
            type=paddle.data_type.integer_value_sequence(target_dict_dim))
        cost = paddle.layer.classification_cost(input=decoder, label=lbl)

        return cost
    else:
        trg_embedding = paddle.layer.GeneratedInput(
            size=target_dict_dim,
            embedding_name='_two_layer_lstm_reader_embedding',
            embedding_size=word_vector_dim)
        group_inputs.append(trg_embedding)

        answer_gen = paddle.layer.beam_search(
            name=decoder_group_name,
            step=two_layer_lstm_decoder,
            input=group_inputs,
            bos_id=0,
            eos_id=1,
            beam_size=beam_size,
            max_length=max_length)

        return answer_gen