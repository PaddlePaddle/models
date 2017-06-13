# coding=utf-8

import paddle.v2 as paddle


def rnn_lm(vocab_size, emb_dim, rnn_type, hidden_size, num_layer):
    """
    RNN language model definition.

    :param vocab_size: size of vocab.
    :param emb_dim: embedding vector's dimension.
    :param rnn_type: the type of RNN cell.
    :param hidden_size: number of unit.
    :param num_layer: layer number.
    :return: cost and output layer of model.
    """

    assert emb_dim > 0 and hidden_size > 0 and vocab_size > 0 and num_layer > 0

    # input layers
    input = paddle.layer.data(
        name="input", type=paddle.data_type.integer_value_sequence(vocab_size))
    target = paddle.layer.data(
        name="target", type=paddle.data_type.integer_value_sequence(vocab_size))

    # embedding layer
    input_emb = paddle.layer.embedding(input=input, size=emb_dim)

    # rnn layer
    if rnn_type == 'lstm':
        rnn_cell = paddle.networks.simple_lstm(
            input=input_emb, size=hidden_size)
        for _ in range(num_layer - 1):
            rnn_cell = paddle.networks.simple_lstm(
                input=rnn_cell, size=hidden_size)
    elif rnn_type == 'gru':
        rnn_cell = paddle.networks.simple_gru(input=input_emb, size=hidden_size)
        for _ in range(num_layer - 1):
            rnn_cell = paddle.networks.simple_gru(
                input=rnn_cell, size=hidden_size)
    else:
        raise Exception('rnn_type error!')

    # fc(full connected) and output layer
    output = paddle.layer.fc(
        input=[rnn_cell], size=vocab_size, act=paddle.activation.Softmax())

    # loss
    cost = paddle.layer.classification_cost(input=output, label=target)

    return cost, output


def ngram_lm(vocab_size, emb_dim, hidden_size, num_layer):
    """
    N-Gram language model definition.

    :param vocab_size: size of vocab.
    :param emb_dim: embedding vector's dimension.
    :param hidden_size: size of unit.
    :param num_layer: layer number.
    :return: cost and output layer of model.
    """

    assert emb_dim > 0 and hidden_size > 0 and vocab_size > 0 and num_layer > 0

    def wordemb(inlayer):
        wordemb = paddle.layer.table_projection(
            input=inlayer,
            size=emb_dim,
            param_attr=paddle.attr.Param(
                name="_proj", initial_std=0.001, learning_rate=1, l2_rate=0))
        return wordemb

    # input layers
    first_word = paddle.layer.data(
        name="first_word", type=paddle.data_type.integer_value(vocab_size))
    second_word = paddle.layer.data(
        name="second_word", type=paddle.data_type.integer_value(vocab_size))
    third_word = paddle.layer.data(
        name="third_word", type=paddle.data_type.integer_value(vocab_size))
    fourth_word = paddle.layer.data(
        name="fourth_word", type=paddle.data_type.integer_value(vocab_size))
    next_word = paddle.layer.data(
        name="next_word", type=paddle.data_type.integer_value(vocab_size))

    # embedding layer
    first_emb = wordemb(first_word)
    second_emb = wordemb(second_word)
    third_emb = wordemb(third_word)
    fourth_emb = wordemb(fourth_word)

    context_emb = paddle.layer.concat(
        input=[first_emb, second_emb, third_emb, fourth_emb])

    # hidden layer
    hidden = paddle.layer.fc(
        input=context_emb, size=hidden_size, act=paddle.activation.Relu())
    for _ in range(num_layer - 1):
        hidden = paddle.layer.fc(
            input=hidden, size=hidden_size, act=paddle.activation.Relu())

    # fc(full connected) and output layer
    predict_word = paddle.layer.fc(
        input=[hidden], size=vocab_size, act=paddle.activation.Softmax())

    # loss
    cost = paddle.layer.classification_cost(input=predict_word, label=next_word)

    return cost, predict_word
