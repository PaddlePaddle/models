import math

import paddle.v2 as paddle
import paddle.v2.evaluator as evaluator


def stacked_rnn(input_layer,
                hidden_size,
                hidden_para_attr,
                rnn_para_attr,
                stack_num=3,
                reverse=False):
    for i in range(stack_num):
        hidden = paddle.layer.fc(
            size=hidden_size,
            act=paddle.activation.Tanh(),
            bias_attr=paddle.attr.Param(initial_std=1.),
            input=[input_layer] if not i else [hidden, rnn],
            param_attr=[rnn_para_attr]
            if not i else [hidden_para_attr, rnn_para_attr])

        rnn = paddle.layer.recurrent(
            input=hidden,
            act=paddle.activation.Relu(),
            bias_attr=paddle.attr.Param(initial_std=1.),
            reverse=reverse,
            param_attr=rnn_para_attr)
    return hidden, rnn


def ner_net(word_dict_len, label_dict_len, stack_num=3, is_train=True):
    mark_dict_len = 2
    word_dim = 50
    mark_dim = 5
    hidden_dim = 128

    word = paddle.layer.data(
        name='word',
        type=paddle.data_type.integer_value_sequence(word_dict_len))
    word_embedding = paddle.layer.embedding(
        input=word,
        size=word_dim,
        param_attr=paddle.attr.Param(
            name='emb', initial_std=math.sqrt(1. / word_dim), is_static=True))

    mark = paddle.layer.data(
        name='mark',
        type=paddle.data_type.integer_value_sequence(mark_dict_len))
    mark_embedding = paddle.layer.embedding(
        input=mark,
        size=mark_dim,
        param_attr=paddle.attr.Param(initial_std=math.sqrt(1. / word_dim)))

    emb_layers = [word_embedding, mark_embedding]

    word_caps_vector = paddle.layer.concat(input=emb_layers)

    mix_hidden_lr = 1e-3
    rnn_para_attr = paddle.attr.Param(initial_std=0.0, learning_rate=0.1)
    hidden_para_attr = paddle.attr.Param(
        initial_std=1 / math.sqrt(hidden_dim), learning_rate=mix_hidden_lr)

    forward_hidden, rnn_forward = stacked_rnn(word_caps_vector, hidden_dim,
                                              hidden_para_attr, rnn_para_attr)
    backward_hidden, rnn_backward = stacked_rnn(
        word_caps_vector,
        hidden_dim,
        hidden_para_attr,
        rnn_para_attr,
        reverse=True)

    fea = paddle.layer.fc(
        size=hidden_dim,
        bias_attr=paddle.attr.Param(initial_std=1.),
        act=paddle.activation.STanh(),
        input=[forward_hidden, rnn_forward, backward_hidden, rnn_backward],
        param_attr=[
            hidden_para_attr, rnn_para_attr, hidden_para_attr, rnn_para_attr
        ])

    emission = paddle.layer.fc(
        size=label_dict_len,
        bias_attr=False,
        input=fea,
        param_attr=rnn_para_attr)

    if is_train:
        target = paddle.layer.data(
            name='target',
            type=paddle.data_type.integer_value_sequence(label_dict_len))

        crf = paddle.layer.crf(
            size=label_dict_len,
            input=emission,
            label=target,
            param_attr=paddle.attr.Param(name='crfw', initial_std=1e-3))

        crf_dec = paddle.layer.crf_decoding(
            size=label_dict_len,
            input=emission,
            label=target,
            param_attr=paddle.attr.Param(name='crfw'))
        return crf, crf_dec, target
    else:
        predict = paddle.layer.crf_decoding(
            size=label_dict_len,
            input=emission,
            param_attr=paddle.attr.Param(name='crfw'))
        return predict
