#coding=utf-8

import math

import paddle.v2 as paddle

__all__ = ["conv_seq2seq"]


def gated_conv_with_batchnorm(input,
                              size,
                              context_len,
                              context_start=None,
                              learning_rate=1.0,
                              drop_rate=0.,
                              with_bn=False):
    """
    Definition of the convolution block.

    :param input: The input of this block.
    :type input: LayerOutput
    :param size: The dimension of the block's output.
    :type size: int
    :param context_len: The context length of the convolution.
    :type context_len: int
    :param context_start: The start position of the context.
    :type context_start: int
    :param learning_rate: The learning rate factor of the parameters in the block.
                          The actual learning rate is the product of the global
                          learning rate and this factor.
    :type learning_rate: float
    :param drop_rate: Dropout rate.
    :type drop_rate: float
    :param with_bn: Whether to use batch normalization or not. False is the default
                    value.
    :type with_bn: bool
    :return: The output of the convolution block.
    :rtype: LayerOutput
    """
    input = paddle.layer.dropout(input=input, dropout_rate=drop_rate)

    context = paddle.layer.mixed(
        size=input.size * context_len,
        input=paddle.layer.context_projection(
            input=input, context_len=context_len, context_start=context_start))

    raw_conv = paddle.layer.fc(
        input=context,
        size=size * 2,
        act=paddle.activation.Linear(),
        param_attr=paddle.attr.Param(
            initial_mean=0.,
            initial_std=math.sqrt(4.0 * (1.0 - drop_rate) / context.size),
            learning_rate=learning_rate),
        bias_attr=False)

    if with_bn:
        raw_conv = paddle.layer.batch_norm(
            input=raw_conv,
            act=paddle.activation.Linear(),
            param_attr=paddle.attr.Param(learning_rate=learning_rate))

    with paddle.layer.mixed(size=size) as conv:
        conv += paddle.layer.identity_projection(raw_conv, size=size, offset=0)

    with paddle.layer.mixed(size=size, act=paddle.activation.Sigmoid()) as gate:
        gate += paddle.layer.identity_projection(
            raw_conv, size=size, offset=size)

    with paddle.layer.mixed(size=size) as gated_conv:
        gated_conv += paddle.layer.dotmul_operator(conv, gate)

    return gated_conv


def encoder(token_emb,
            pos_emb,
            conv_blocks=[(256, 3)] * 5,
            num_attention=3,
            drop_rate=0.,
            with_bn=False):
    """
    Definition of the encoder.

    :param token_emb: The embedding vector of the input token.
    :type token_emb: LayerOutput
    :param pos_emb: The embedding vector of the input token's position.
    :type pos_emb: LayerOutput
    :param conv_blocks: The scale list of the convolution blocks. Each element of
                        the list contains output dimension and context length of
                        the corresponding convolution block.
    :type conv_blocks: list of tuple
    :param num_attention: The total number of the attention modules used in the decoder.
    :type num_attention: int
    :param drop_rate: Dropout rate.
    :type drop_rate: float
    :param with_bn: Whether to use batch normalization or not. False is the default
                    value.
    :type with_bn: bool
    :return: The input token encoding.
    :rtype: LayerOutput
    """
    embedding = paddle.layer.addto(
        input=[token_emb, pos_emb],
        layer_attr=paddle.attr.Extra(drop_rate=drop_rate))

    proj_size = conv_blocks[0][0]
    block_input = paddle.layer.fc(
        input=embedding,
        size=proj_size,
        act=paddle.activation.Linear(),
        param_attr=paddle.attr.Param(
            initial_mean=0.,
            initial_std=math.sqrt((1.0 - drop_rate) / embedding.size),
            learning_rate=1.0 / (2.0 * num_attention)),
        bias_attr=True, )

    for (size, context_len) in conv_blocks:
        if block_input.size == size:
            residual = block_input
        else:
            residual = paddle.layer.fc(
                input=block_input,
                size=size,
                act=paddle.activation.Linear(),
                param_attr=paddle.attr.Param(learning_rate=1.0 /
                                             (2.0 * num_attention)),
                bias_attr=True)

        gated_conv = gated_conv_with_batchnorm(
            input=block_input,
            size=size,
            context_len=context_len,
            learning_rate=1.0 / (2.0 * num_attention),
            drop_rate=drop_rate,
            with_bn=with_bn)

        with paddle.layer.mixed(size=size) as block_output:
            block_output += paddle.layer.identity_projection(residual)
            block_output += paddle.layer.identity_projection(gated_conv)

        # halve the variance of the sum
        block_output = paddle.layer.slope_intercept(
            input=block_output, slope=math.sqrt(0.5))

        block_input = block_output

    emb_dim = embedding.size
    encoded_vec = paddle.layer.fc(
        input=block_output,
        size=emb_dim,
        act=paddle.activation.Linear(),
        param_attr=paddle.attr.Param(learning_rate=1.0 / (2.0 * num_attention)),
        bias_attr=True)

    encoded_sum = paddle.layer.addto(input=[encoded_vec, embedding])

    # halve the variance of the sum
    encoded_sum = paddle.layer.slope_intercept(
        input=encoded_sum, slope=math.sqrt(0.5))

    return encoded_vec, encoded_sum


def attention(decoder_state, cur_embedding, encoded_vec, encoded_sum):
    """
    Definition of the attention.

    :param decoder_state: The hidden state of the decoder.
    :type decoder_state: LayerOutput
    :param cur_embedding: The embedding vector of the current token.
    :type cur_embedding: LayerOutput
    :param encoded_vec: The source token encoding.
    :type encoded_vec: LayerOutput
    :param encoded_sum: The sum of the source token's encoding and embedding.
    :type encoded_sum: LayerOutput
    :return: A context vector and the attention weight.
    :rtype: LayerOutput
    """
    residual = decoder_state

    state_size = decoder_state.size
    emb_dim = cur_embedding.size
    with paddle.layer.mixed(size=emb_dim, bias_attr=True) as state_summary:
        state_summary += paddle.layer.full_matrix_projection(decoder_state)
        state_summary += paddle.layer.identity_projection(cur_embedding)

    # halve the variance of the sum
    state_summary = paddle.layer.slope_intercept(
        input=state_summary, slope=math.sqrt(0.5))

    expanded = paddle.layer.expand(input=state_summary, expand_as=encoded_vec)

    m = paddle.layer.dot_prod(input1=expanded, input2=encoded_vec)

    attention_weight = paddle.layer.fc(input=m,
                                       size=1,
                                       act=paddle.activation.SequenceSoftmax(),
                                       bias_attr=False)

    scaled = paddle.layer.scaling(weight=attention_weight, input=encoded_sum)

    attended = paddle.layer.pooling(
        input=scaled, pooling_type=paddle.pooling.Sum())

    attended_proj = paddle.layer.fc(input=attended,
                                    size=state_size,
                                    act=paddle.activation.Linear(),
                                    bias_attr=True)

    attention_result = paddle.layer.addto(input=[attended_proj, residual])

    # halve the variance of the sum
    attention_result = paddle.layer.slope_intercept(
        input=attention_result, slope=math.sqrt(0.5))
    return attention_result, attention_weight


def decoder(token_emb,
            pos_emb,
            encoded_vec,
            encoded_sum,
            dict_size,
            conv_blocks=[(256, 3)] * 3,
            drop_rate=0.,
            with_bn=False):
    """
    Definition of the decoder.

    :param token_emb: The embedding vector of the input token.
    :type token_emb: LayerOutput
    :param pos_emb: The embedding vector of the input token's position.
    :type pos_emb: LayerOutput
    :param encoded_vec: The source token encoding.
    :type encoded_vec: LayerOutput
    :param encoded_sum: The sum of the source token's encoding and embedding.
    :type encoded_sum: LayerOutput
    :param dict_size: The size of the target dictionary.
    :type dict_size: int
    :param conv_blocks: The scale list of the convolution blocks. Each element
                        of the list contains output dimension and context length
                        of the corresponding convolution block.
    :type conv_blocks: list of tuple
    :param drop_rate: Dropout rate.
    :type drop_rate: float
    :param with_bn: Whether to use batch normalization or not. False is the default
                    value.
    :type with_bn: bool
    :return: The probability of the predicted token and the attention weights.
    :rtype: LayerOutput
    """

    def attention_step(decoder_state, cur_embedding, encoded_vec, encoded_sum):
        conditional = attention(
            decoder_state=decoder_state,
            cur_embedding=cur_embedding,
            encoded_vec=encoded_vec,
            encoded_sum=encoded_sum)
        return conditional

    embedding = paddle.layer.addto(
        input=[token_emb, pos_emb],
        layer_attr=paddle.attr.Extra(drop_rate=drop_rate))

    proj_size = conv_blocks[0][0]
    block_input = paddle.layer.fc(
        input=embedding,
        size=proj_size,
        act=paddle.activation.Linear(),
        param_attr=paddle.attr.Param(
            initial_mean=0.,
            initial_std=math.sqrt((1.0 - drop_rate) / embedding.size)),
        bias_attr=True, )

    weight = []
    for (size, context_len) in conv_blocks:
        if block_input.size == size:
            residual = block_input
        else:
            residual = paddle.layer.fc(input=block_input,
                                       size=size,
                                       act=paddle.activation.Linear(),
                                       bias_attr=True)

        decoder_state = gated_conv_with_batchnorm(
            input=block_input,
            size=size,
            context_len=context_len,
            context_start=0,
            drop_rate=drop_rate,
            with_bn=with_bn)

        group_inputs = [
            decoder_state,
            embedding,
            paddle.layer.StaticInput(input=encoded_vec),
            paddle.layer.StaticInput(input=encoded_sum),
        ]

        conditional, attention_weight = paddle.layer.recurrent_group(
            step=attention_step, input=group_inputs)
        weight.append(attention_weight)

        block_output = paddle.layer.addto(input=[conditional, residual])

        # halve the variance of the sum
        block_output = paddle.layer.slope_intercept(
            input=block_output, slope=math.sqrt(0.5))

        block_input = block_output

    out_emb_dim = embedding.size
    block_output = paddle.layer.fc(
        input=block_output,
        size=out_emb_dim,
        act=paddle.activation.Linear(),
        layer_attr=paddle.attr.Extra(drop_rate=drop_rate))

    decoder_out = paddle.layer.fc(
        input=block_output,
        size=dict_size,
        act=paddle.activation.Softmax(),
        param_attr=paddle.attr.Param(
            initial_mean=0.,
            initial_std=math.sqrt((1.0 - drop_rate) / block_output.size)),
        bias_attr=True)

    return decoder_out, weight


def conv_seq2seq(src_dict_size,
                 trg_dict_size,
                 pos_size,
                 emb_dim,
                 enc_conv_blocks=[(256, 3)] * 5,
                 dec_conv_blocks=[(256, 3)] * 3,
                 drop_rate=0.,
                 with_bn=False,
                 is_infer=False):
    """
    Definition of convolutional sequence-to-sequence network.

    :param src_dict_size: The size of the source dictionary.
    :type src_dict_size: int
    :param trg_dict_size: The size of the target dictionary.
    :type trg_dict_size: int
    :param pos_size: The total number of the position indexes, which means
                     the maximum value of the index is pos_size - 1.
    :type pos_size: int
    :param emb_dim: The dimension of the embedding vector.
    :type emb_dim: int
    :param enc_conv_blocks: The scale list of the encoder's convolution blocks. Each element
                            of the list contains output dimension and context length of the
                            corresponding convolution block.
    :type enc_conv_blocks: list of tuple
    :param dec_conv_blocks: The scale list of the decoder's convolution blocks. Each element
                            of the list contains output dimension and context length of the
                            corresponding convolution block.
    :type dec_conv_blocks: list of tuple
    :param drop_rate: Dropout rate.
    :type drop_rate: float
    :param with_bn: Whether to use batch normalization or not. False is the default value.
    :type with_bn: bool
    :param is_infer: Whether infer or not.
    :type is_infer: bool
    :return: Cost or output layer.
    :rtype: LayerOutput
    """
    src = paddle.layer.data(
        name='src_word',
        type=paddle.data_type.integer_value_sequence(src_dict_size))
    src_pos = paddle.layer.data(
        name='src_word_pos',
        type=paddle.data_type.integer_value_sequence(pos_size +
                                                     1))  # one for padding

    src_emb = paddle.layer.embedding(
        input=src,
        size=emb_dim,
        name='src_word_emb',
        param_attr=paddle.attr.Param(
            initial_mean=0., initial_std=0.1))
    src_pos_emb = paddle.layer.embedding(
        input=src_pos,
        size=emb_dim,
        name='src_pos_emb',
        param_attr=paddle.attr.Param(
            initial_mean=0., initial_std=0.1))

    num_attention = len(dec_conv_blocks)
    encoded_vec, encoded_sum = encoder(
        token_emb=src_emb,
        pos_emb=src_pos_emb,
        conv_blocks=enc_conv_blocks,
        num_attention=num_attention,
        drop_rate=drop_rate,
        with_bn=with_bn)

    trg = paddle.layer.data(
        name='trg_word',
        type=paddle.data_type.integer_value_sequence(trg_dict_size +
                                                     1))  # one for padding
    trg_pos = paddle.layer.data(
        name='trg_word_pos',
        type=paddle.data_type.integer_value_sequence(pos_size +
                                                     1))  # one for padding

    trg_emb = paddle.layer.embedding(
        input=trg,
        size=emb_dim,
        name='trg_word_emb',
        param_attr=paddle.attr.Param(
            initial_mean=0., initial_std=0.1))
    trg_pos_emb = paddle.layer.embedding(
        input=trg_pos,
        size=emb_dim,
        name='trg_pos_emb',
        param_attr=paddle.attr.Param(
            initial_mean=0., initial_std=0.1))

    decoder_out, weight = decoder(
        token_emb=trg_emb,
        pos_emb=trg_pos_emb,
        encoded_vec=encoded_vec,
        encoded_sum=encoded_sum,
        dict_size=trg_dict_size,
        conv_blocks=dec_conv_blocks,
        drop_rate=drop_rate,
        with_bn=with_bn)

    if is_infer:
        return decoder_out, weight

    trg_next_word = paddle.layer.data(
        name='trg_next_word',
        type=paddle.data_type.integer_value_sequence(trg_dict_size))
    cost = paddle.layer.classification_cost(
        input=decoder_out, label=trg_next_word)

    return cost
