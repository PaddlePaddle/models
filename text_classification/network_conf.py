import sys
import math
import gzip

from paddle.v2.layer import parse_network
import paddle.v2 as paddle

__all__ = ["fc_net", "convolution_net"]


def fc_net(dict_dim,
           class_num,
           emb_dim=28,
           hidden_layer_sizes=[28, 8],
           is_infer=False):
    """
    define the topology of the dnn network

    :param dict_dim: size of word dictionary
    :type input_dim: int
    :params class_num: number of instance class
    :type class_num: int
    :params emb_dim: embedding vector dimension
    :type emb_dim: int
    """

    # define the input layers
    data = paddle.layer.data("word",
                             paddle.data_type.integer_value_sequence(dict_dim))
    if not is_infer:
        lbl = paddle.layer.data("label",
                                paddle.data_type.integer_value(class_num))

    # define the embedding layer
    emb = paddle.layer.embedding(input=data, size=emb_dim)
    # max pooling to reduce the input sequence into a vector (non-sequence)
    seq_pool = paddle.layer.pooling(
        input=emb, pooling_type=paddle.pooling.Max())

    for idx, hidden_size in enumerate(hidden_layer_sizes):
        hidden_init_std = 1.0 / math.sqrt(hidden_size)
        hidden = paddle.layer.fc(
            input=hidden if idx else seq_pool,
            size=hidden_size,
            act=paddle.activation.Tanh(),
            param_attr=paddle.attr.Param(initial_std=hidden_init_std))

    prob = paddle.layer.fc(
        input=hidden,
        size=class_num,
        act=paddle.activation.Softmax(),
        param_attr=paddle.attr.Param(initial_std=1.0 / math.sqrt(class_num)))

    if is_infer:
        return prob
    else:
        return paddle.layer.classification_cost(
            input=prob, label=lbl), prob, lbl


def convolution_net(dict_dim,
                    class_dim=2,
                    emb_dim=28,
                    hid_dim=128,
                    is_infer=False):
    """
    cnn network definition

    :param dict_dim: size of word dictionary
    :type input_dim: int
    :params class_dim: number of instance class
    :type class_dim: int
    :params emb_dim: embedding vector dimension
    :type emb_dim: int
    :params hid_dim: number of same size convolution kernels
    :type hid_dim: int
    """

    # input layers
    data = paddle.layer.data("word",
                             paddle.data_type.integer_value_sequence(dict_dim))
    lbl = paddle.layer.data("label", paddle.data_type.integer_value(class_dim))

    # embedding layer
    emb = paddle.layer.embedding(input=data, size=emb_dim)

    # convolution layers with max pooling
    conv_3 = paddle.networks.sequence_conv_pool(
        input=emb, context_len=3, hidden_size=hid_dim)
    conv_4 = paddle.networks.sequence_conv_pool(
        input=emb, context_len=4, hidden_size=hid_dim)

    # fc and output layer
    prob = paddle.layer.fc(input=[conv_3, conv_4],
                           size=class_dim,
                           act=paddle.activation.Softmax())

    if is_infer:
        return prob
    else:
        cost = paddle.layer.classification_cost(input=prob, label=lbl)

        return cost, prob, lbl
