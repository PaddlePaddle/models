# coding=utf-8

import paddle.v2 as paddle


def rnn_lm(vocab_size,
           emb_dim,
           rnn_type,
           hidden_size,
           num_layer,
           is_infer=False):
    """
    RNN language model definition.

    :param vocab_size: size of vocab.
    :param emb_dim: embedding vector"s dimension.
    :param rnn_type: the type of RNN cell.
    :param hidden_size: number of unit.
    :param num_layer: layer number.
    :return: cost and output layer of model.
    """

    # input layers
    input = paddle.layer.data(
        name="input", type=paddle.data_type.integer_value_sequence(vocab_size))
    if not is_infer:
        target = paddle.layer.data(
            name="target",
            type=paddle.data_type.integer_value_sequence(vocab_size))

    # embedding layer
    input_emb = paddle.layer.embedding(input=input, size=emb_dim)

    # rnn layer
    if rnn_type == "lstm":
        rnn_cell = paddle.networks.simple_lstm(
            input=input_emb, size=hidden_size)
        for _ in range(num_layer - 1):
            rnn_cell = paddle.networks.simple_lstm(
                input=rnn_cell, size=hidden_size)
    elif rnn_type == "gru":
        rnn_cell = paddle.networks.simple_gru(input=input_emb, size=hidden_size)
        for _ in range(num_layer - 1):
            rnn_cell = paddle.networks.simple_gru(
                input=rnn_cell, size=hidden_size)
    else:
        raise Exception("rnn_type error!")

    # fc(full connected) and output layer
    output = paddle.layer.fc(
        input=[rnn_cell], size=vocab_size, act=paddle.activation.Softmax())

    if is_infer:
        return output
    else:
        cost = paddle.layer.classification_cost(input=output, label=target)

        return cost, output


def ngram_lm(vocab_size,
             emb_dim,
             hidden_size,
             num_layer,
             gram_num,
             is_infer=False):
    """
    N-Gram language model definition.

    :param vocab_size: size of vocab.
    :param emb_dim: embedding vector"s dimension.
    :param hidden_size: size of unit.
    :param num_layer: number of hidden layers.
    :param gram_num: gram number in n-gram method
    :return: cost and output layer of model.
    """

    # input layers
    emb_layers = []
    for i in range(gram_num):
        word = paddle.layer.data(
            name="__word%02d__" % (i + 1),
            type=paddle.data_type.integer_value(vocab_size))
        emb = paddle.layer.embedding(
            input=word,
            size=emb_dim,
            param_attr=paddle.attr.Param(name="_proj", initial_std=1e-3))
        emb_layers.append(emb)
    if not is_infer:
        next_word = paddle.layer.data(
            name="__next_word__",
            type=paddle.data_type.integer_value(vocab_size))

    # hidden layer
    for i in range(num_layer):
        hidden = paddle.layer.fc(
            input=hidden if i else paddle.layer.concat(input=emb_layers),
            size=hidden_size,
            act=paddle.activation.Relu())

    predict_word = paddle.layer.fc(
        input=[hidden], size=vocab_size, act=paddle.activation.Softmax())

    if is_infer:
        return predict_word
    else:
        cost = paddle.layer.classification_cost(
            input=predict_word, label=next_word)

        return cost, predict_word
