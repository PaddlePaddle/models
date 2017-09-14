import math

import paddle.v2 as paddle
import paddle.v2.evaluator as evaluator


def ner_net(word_dict_len, label_dict_len, stack_num=2, is_train=True):
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

    word_caps_vector = paddle.layer.concat(
        input=[word_embedding, mark_embedding])

    mix_hidden_lr = 1e-3
    rnn_para_attr = paddle.attr.Param(initial_std=0.0, learning_rate=0.1)
    hidden_para_attr = paddle.attr.Param(
        initial_std=1 / math.sqrt(hidden_dim), learning_rate=mix_hidden_lr)

    # the first rnn layer shares the input-to-hidden mappings.
    hidden = paddle.layer.fc(
        name="__hidden00__",
        size=hidden_dim,
        act=paddle.activation.Tanh(),
        bias_attr=paddle.attr.Param(initial_std=1.),
        input=word_caps_vector,
        param_attr=hidden_para_attr)

    fea = []
    for direction in ["fwd", "bwd"]:
        for i in range(stack_num):
            if i:
                hidden = paddle.layer.fc(
                    name="__hidden%02d_%s__" % (i, direction),
                    size=hidden_dim,
                    act=paddle.activation.STanh(),
                    bias_attr=paddle.attr.Param(initial_std=1.),
                    input=[hidden, rnn],
                    param_attr=[hidden_para_attr, rnn_para_attr])

            rnn = paddle.layer.recurrent(
                name="__rnn%02d_%s__" % (i, direction),
                input=hidden,
                act=paddle.activation.Relu(),
                bias_attr=paddle.attr.Param(initial_std=1.),
                reverse=i % 2 if direction == "fwd" else not i % 2,
                param_attr=rnn_para_attr)
        fea += [hidden, rnn]

    rnn_fea = paddle.layer.fc(
        size=hidden_dim,
        bias_attr=paddle.attr.Param(initial_std=1.),
        act=paddle.activation.STanh(),
        input=fea,
        param_attr=[hidden_para_attr, rnn_para_attr] * 2)

    emission = paddle.layer.fc(
        size=label_dict_len,
        bias_attr=False,
        input=rnn_fea,
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
