import paddle.v2 as paddle
import sys
import gzip


def seq2seq_net(source_dict_dim,
                target_dict_dim,
                word_vector_dim=620,
                rnn_hidden_size=1000,
                beam_size=1,
                max_length=50,
                is_generating=False):
    """
    Define the network structure of NMT, including encoder and decoder.

    :param source_dict_dim: size of source dictionary
    :type source_dict_dim : int
    :param target_dict_dim: size of target dictionary
    :type target_dict_dim: int
    :param word_vector_dim: size of source language word embedding
    :type word_vector_dim: int
    :param rnn_hidden_size: size of hidden state of encoder and decoder RNN
    :type rnn_hidden_size: int
    :param beam_size: expansion width in each step when generating
    :type beam_size: int
    :param max_length: max iteration number in generation
    :type max_length: int
    :param generating: whether to generate sequence or to train
    :type generating: bool
    """

    decoder_size = encoder_size = rnn_hidden_size

    src_word_id = paddle.layer.data(
        name="source_language_word",
        type=paddle.data_type.integer_value_sequence(source_dict_dim))
    src_embedding = paddle.layer.embedding(
        input=src_word_id, size=word_vector_dim)

    # use bidirectional_gru as the encoder
    encoded_vector = paddle.networks.bidirectional_gru(
        input=src_embedding,
        size=encoder_size,
        fwd_act=paddle.activation.Tanh(),
        fwd_gate_act=paddle.activation.Sigmoid(),
        bwd_act=paddle.activation.Tanh(),
        bwd_gate_act=paddle.activation.Sigmoid(),
        return_seq=True)
    #### Decoder
    encoder_last = paddle.layer.last_seq(input=encoded_vector)
    encoder_last_projected = paddle.layer.fc(size=decoder_size,
                                             act=paddle.activation.Tanh(),
                                             input=encoder_last)

    # gru step
    def gru_decoder_without_attention(enc_vec, current_word):
        """
        Step function for gru decoder

        :param enc_vec: encoded vector of source language
        :type enc_vec: layer object
        :param current_word: current input of decoder
        :type current_word: layer object
        """
        decoder_mem = paddle.layer.memory(
            name="gru_decoder",
            size=decoder_size,
            boot_layer=encoder_last_projected)

        context = paddle.layer.last_seq(input=enc_vec)

        decoder_inputs = paddle.layer.fc(size=decoder_size * 3,
                                         input=[context, current_word])

        gru_step = paddle.layer.gru_step(
            name="gru_decoder",
            act=paddle.activation.Tanh(),
            gate_act=paddle.activation.Sigmoid(),
            input=decoder_inputs,
            output_mem=decoder_mem,
            size=decoder_size)

        out = paddle.layer.fc(size=target_dict_dim,
                              bias_attr=True,
                              act=paddle.activation.Softmax(),
                              input=gru_step)
        return out

    group_input1 = paddle.layer.StaticInput(input=encoded_vector)
    group_inputs = [group_input1]

    decoder_group_name = "decoder_group"
    if is_generating:
        trg_embedding = paddle.layer.GeneratedInput(
            size=target_dict_dim,
            embedding_name="_target_language_embedding",
            embedding_size=word_vector_dim)
        group_inputs.append(trg_embedding)

        beam_gen = paddle.layer.beam_search(
            name=decoder_group_name,
            step=gru_decoder_without_attention,
            input=group_inputs,
            bos_id=0,
            eos_id=1,
            beam_size=beam_size,
            max_length=max_length)

        return beam_gen
    else:
        trg_embedding = paddle.layer.embedding(
            input=paddle.layer.data(
                name="target_language_word",
                type=paddle.data_type.integer_value_sequence(target_dict_dim)),
            size=word_vector_dim,
            param_attr=paddle.attr.ParamAttr(name="_target_language_embedding"))
        group_inputs.append(trg_embedding)

        decoder = paddle.layer.recurrent_group(
            name=decoder_group_name,
            step=gru_decoder_without_attention,
            input=group_inputs)

        lbl = paddle.layer.data(
            name="target_language_next_word",
            type=paddle.data_type.integer_value_sequence(target_dict_dim))
        cost = paddle.layer.classification_cost(input=decoder, label=lbl)

        return cost
