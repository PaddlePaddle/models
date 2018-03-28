import math

import paddle.v2 as paddle
from paddle.v2.layer import parse_network


def ngram_lm(hidden_size, emb_size, dict_size, gram_num=4, is_train=True):
    emb_layers = []
    embed_param_attr = paddle.attr.Param(
        name="_proj", initial_std=0.001, learning_rate=1, l2_rate=0)
    for i in range(gram_num):
        word = paddle.layer.data(
            name="__word%02d__" % (i),
            type=paddle.data_type.integer_value(dict_size))
        emb_layers.append(
            paddle.layer.embedding(
                input=word, size=emb_size, param_attr=embed_param_attr))
    next_word = paddle.layer.data(
        name="__target_word__", type=paddle.data_type.integer_value(dict_size))

    context_embedding = paddle.layer.concat(input=emb_layers)

    hidden_layer = paddle.layer.fc(
        input=context_embedding,
        size=hidden_size,
        act=paddle.activation.Tanh(),
        param_attr=paddle.attr.Param(initial_std=1. / math.sqrt(emb_size * 8)))

    if is_train:
        return paddle.layer.nce(input=hidden_layer,
                                label=next_word,
                                num_classes=dict_size,
                                param_attr=paddle.attr.Param(name="nce_w"),
                                bias_attr=paddle.attr.Param(name="nce_b"),
                                num_neg_samples=25,
                                neg_distribution=None)
    else:
        return paddle.layer.mixed(
            size=dict_size,
            input=paddle.layer.trans_full_matrix_projection(
                hidden_layer, param_attr=paddle.attr.Param(name="nce_w")),
            act=paddle.activation.Softmax(),
            bias_attr=paddle.attr.Param(name="nce_b"))


if __name__ == "__main__":
    # this is to test and debug the network topology defination.
    # please set the hyper-parameters as needed.
    print(parse_network(
        ngram_lm(
            hidden_size=256,
            emb_size=256,
            dict_size=1024,
            gram_num=4,
            is_train=True)))
