import paddle.v2 as paddle


def rnn_lm(vocab_dim,
           emb_dim,
           hidden_size,
           stacked_rnn_num,
           rnn_type="lstm",
           is_infer=False):
    """
    RNN language model definition.

    :param vocab_dim: size of vocabulary.
    :type vocab_dim: int
    :param emb_dim: dimension of the embedding vector
    :type emb_dim: int
    :param rnn_type: the type of RNN cell.
    :type rnn_type: int
    :param hidden_size: number of hidden unit.
    :type hidden_size: int
    :param stacked_rnn_num: number of stacked rnn cell.
    :type stacked_rnn_num: int
    :return: cost and output layer of model.
    :rtype: LayerOutput
    """

    # input layers
    input = paddle.layer.data(
        name="input", type=paddle.data_type.integer_value_sequence(vocab_dim))
    if not is_infer:
        target = paddle.layer.data(
            name="target",
            type=paddle.data_type.integer_value_sequence(vocab_dim))

    # embedding layer
    input_emb = paddle.layer.embedding(input=input, size=emb_dim)

    # rnn layer
    if rnn_type == "lstm":
        for i in range(stacked_rnn_num):
            rnn_cell = paddle.networks.simple_lstm(
                input=rnn_cell if i else input_emb, size=hidden_size)
    elif rnn_type == "gru":
        for i in range(stacked_rnn_num):
            rnn_cell = paddle.networks.simple_gru(
                input=rnn_cell if i else input_emb, size=hidden_size)
    else:
        raise Exception("rnn_type error!")

    # fc(full connected) and output layer
    output = paddle.layer.fc(input=[rnn_cell],
                             size=vocab_dim,
                             act=paddle.activation.Softmax())

    if is_infer:
        last_word = paddle.layer.last_seq(input=output)
        return last_word
    else:
        cost = paddle.layer.classification_cost(input=output, label=target)

        return cost
