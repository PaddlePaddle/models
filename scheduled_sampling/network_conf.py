import paddle.v2 as paddle

__all__ = ["seqToseq_net"]

### Network Architecture
word_vector_dim = 512  # dimension of word vector
decoder_size = 512  # dimension of hidden unit in GRU Decoder network
encoder_size = 512  # dimension of hidden unit in GRU Encoder network

max_length = 250


def seqToseq_net(source_dict_dim,
                 target_dict_dim,
                 beam_size,
                 is_generating=False):
    """
    The definition of the sequence to sequence model
    :param source_dict_dim: the dictionary size of the source language
    :type source_dict_dim: int
    :param target_dict_dim: the dictionary size of the target language
    :type target_dict_dim: int
    :param beam_size: The width of beam expansion
    :type beam_size: int
    :param is_generating: whether in generating mode
    :type is_generating: Bool
    :return: the last layer of the network
    :rtype: LayerOutput
    """

    #### Encoder
    src_word_id = paddle.layer.data(
        name='source_language_word',
        type=paddle.data_type.integer_value_sequence(source_dict_dim))
    src_embedding = paddle.layer.embedding(
        input=src_word_id, size=word_vector_dim)
    src_forward = paddle.networks.simple_gru(
        input=src_embedding, size=encoder_size)
    src_reverse = paddle.networks.simple_gru(
        input=src_embedding, size=encoder_size, reverse=True)
    encoded_vector = paddle.layer.concat(input=[src_forward, src_reverse])

    #### Decoder
    encoded_proj = paddle.layer.fc(input=encoded_vector,
                                   size=decoder_size,
                                   act=paddle.activation.Linear(),
                                   bias_attr=False)

    reverse_first = paddle.layer.first_seq(input=src_reverse)

    decoder_boot = paddle.layer.fc(input=reverse_first,
                                   size=decoder_size,
                                   act=paddle.activation.Tanh(),
                                   bias_attr=False)

    def gru_decoder_with_attention_train(enc_vec, enc_proj, true_word,
                                         true_token_flag):
        """
        The decoder step for training.
        :param enc_vec: the encoder vector for attention
        :type enc_vec: LayerOutput
        :param enc_proj: the encoder projection for attention
        :type enc_proj: LayerOutput
        :param true_word: the ground-truth target word
        :type true_word: LayerOutput
        :param true_token_flag: the flag of using the ground-truth target word
        :type true_token_flag: LayerOutput
        :return: the softmax output layer
        :rtype: LayerOutput
        """

        decoder_mem = paddle.layer.memory(
            name='gru_decoder', size=decoder_size, boot_layer=decoder_boot)

        context = paddle.networks.simple_attention(
            encoded_sequence=enc_vec,
            encoded_proj=enc_proj,
            decoder_state=decoder_mem)

        gru_out_memory = paddle.layer.memory(
            name='gru_out', size=target_dict_dim)

        generated_word = paddle.layer.max_id(input=gru_out_memory)

        generated_word_emb = paddle.layer.embedding(
            input=generated_word,
            size=word_vector_dim,
            param_attr=paddle.attr.ParamAttr(name='_target_language_embedding'))

        current_word = paddle.layer.multiplex(
            input=[true_token_flag, true_word, generated_word_emb])

        decoder_inputs = paddle.layer.fc(input=[context, current_word],
                                         size=decoder_size * 3,
                                         act=paddle.activation.Linear(),
                                         bias_attr=False)

        gru_step = paddle.layer.gru_step(
            name='gru_decoder',
            input=decoder_inputs,
            output_mem=decoder_mem,
            size=decoder_size)

        out = paddle.layer.fc(name='gru_out',
                              input=gru_step,
                              size=target_dict_dim,
                              act=paddle.activation.Softmax())
        return out

    def gru_decoder_with_attention_gen(enc_vec, enc_proj, current_word):
        """
        The decoder step for generating.
        :param enc_vec: the encoder vector for attention
        :type enc_vec: LayerOutput
        :param enc_proj: the encoder projection for attention
        :type enc_proj: LayerOutput
        :param current_word: the previously generated word
        :type current_word: LayerOutput
        :return: the softmax output layer
        :rtype: LayerOutput
        """

        decoder_mem = paddle.layer.memory(
            name='gru_decoder', size=decoder_size, boot_layer=decoder_boot)

        context = paddle.networks.simple_attention(
            encoded_sequence=enc_vec,
            encoded_proj=enc_proj,
            decoder_state=decoder_mem)

        decoder_inputs = paddle.layer.fc(input=[context, current_word],
                                         size=decoder_size * 3,
                                         act=paddle.activation.Linear(),
                                         bias_attr=False)

        gru_step = paddle.layer.gru_step(
            name='gru_decoder',
            input=decoder_inputs,
            output_mem=decoder_mem,
            size=decoder_size)

        out = paddle.layer.fc(name='gru_out',
                              input=gru_step,
                              size=target_dict_dim,
                              act=paddle.activation.Softmax())
        return out

    decoder_group_name = "decoder_group"
    group_input1 = paddle.layer.StaticInput(input=encoded_vector, is_seq=True)
    group_input2 = paddle.layer.StaticInput(input=encoded_proj, is_seq=True)

    if not is_generating:
        trg_embedding = paddle.layer.embedding(
            input=paddle.layer.data(
                name='target_language_word',
                type=paddle.data_type.integer_value_sequence(target_dict_dim)),
            size=word_vector_dim,
            param_attr=paddle.attr.ParamAttr(name='_target_language_embedding'))

        true_token_flags = paddle.layer.data(
            name='true_token_flag',
            type=paddle.data_type.integer_value_sequence(2))

        group_inputs = [
            group_input1, group_input2, trg_embedding, true_token_flags
        ]

        decoder = paddle.layer.recurrent_group(
            name=decoder_group_name,
            step=gru_decoder_with_attention_train,
            input=group_inputs)

        lbl = paddle.layer.data(
            name='target_language_next_word',
            type=paddle.data_type.integer_value_sequence(target_dict_dim))

        cost = paddle.layer.classification_cost(input=decoder, label=lbl)

        return cost
    else:
        trg_embedding = paddle.layer.GeneratedInput(
            size=target_dict_dim,
            embedding_name='_target_language_embedding',
            embedding_size=word_vector_dim)

        group_inputs = [group_input1, group_input2, trg_embedding]

        beam_gen = paddle.layer.beam_search(
            name=decoder_group_name,
            step=gru_decoder_with_attention_gen,
            input=group_inputs,
            bos_id=0,
            eos_id=1,
            beam_size=beam_size,
            max_length=max_length)
        return beam_gen
