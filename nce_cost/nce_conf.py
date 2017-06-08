# -*- encoding:utf-8 -*-
import math
import paddle.v2 as paddle


def network_conf(hidden_size, embedding_size, dict_size, is_train):

    first_word = paddle.layer.data(
        name="firstw", type=paddle.data_type.integer_value(dict_size))
    second_word = paddle.layer.data(
        name="secondw", type=paddle.data_type.integer_value(dict_size))
    third_word = paddle.layer.data(
        name="thirdw", type=paddle.data_type.integer_value(dict_size))
    fourth_word = paddle.layer.data(
        name="fourthw", type=paddle.data_type.integer_value(dict_size))
    next_word = paddle.layer.data(
        name="fifthw", type=paddle.data_type.integer_value(dict_size))

    embed_param_attr = paddle.attr.Param(
        name="_proj", initial_std=0.001, learning_rate=1, l2_rate=0)
    first_embedding = paddle.layer.embedding(
        input=first_word, size=embedding_size, param_attr=embed_param_attr)
    second_embedding = paddle.layer.embedding(
        input=second_word, size=embedding_size, param_attr=embed_param_attr)
    third_embedding = paddle.layer.embedding(
        input=third_word, size=embedding_size, param_attr=embed_param_attr)
    fourth_embedding = paddle.layer.embedding(
        input=fourth_word, size=embedding_size, param_attr=embed_param_attr)

    context_embedding = paddle.layer.concat(input=[
        first_embedding, second_embedding, third_embedding, fourth_embedding
    ])

    hidden_layer = paddle.layer.fc(
        input=context_embedding,
        size=hidden_size,
        act=paddle.activation.Tanh(),
        layer_attr=paddle.attr.Extra(drop_rate=0.5),
        bias_attr=paddle.attr.Param(learning_rate=1),
        param_attr=paddle.attr.Param(
            initial_std=1. / math.sqrt(embedding_size * 8), learning_rate=1))

    if is_train == True:
        cost = paddle.layer.nce(
            input=hidden_layer,
            label=next_word,
            num_classes=dict_size,
            act=paddle.activation.Sigmoid(),
            num_neg_samples=25,
            neg_distribution=None,
            param_attr=paddle.attr.Param(name='nce_w'),
            bias_attr=paddle.attr.Param(name='nce_b'))
        return cost
    else:
        with paddle.layer.mixed(
                size=dict_size,
                act=paddle.activation.Softmax(),
                bias_attr=paddle.attr.Param(
                    name='nce_b')) as prediction:
            prediction += paddle.layer.trans_full_matrix_projection(
                input=hidden_layer,
                param_attr=paddle.attr.Param(name='nce_w'))

        return prediction
