import sys
import math
import gzip

from paddle.v2.layer import parse_network
import paddle.v2 as paddle

__all__ = ["cnn_network"]


def cnn_network(dict_dim,
        class_dim=2,
        emb_dim=28,
        hid_dim=128,
        is_infer=False):
 
    data = paddle.layer.data("word",
                             paddle.data_type.integer_value_sequence(dict_dim))
    lbl = paddle.layer.data("label", paddle.data_type.integer_value(class_dim))

    emb = paddle.layer.embedding(input=data, size=emb_dim)

    conv_3 = paddle.networks.sequence_conv_pool(
        input=emb, context_len=3, hidden_size=hid_dim)
    conv_4 = paddle.networks.sequence_conv_pool(
        input=emb, context_len=4, hidden_size=hid_dim)

    prob = paddle.layer.fc(input=[conv_3, conv_4],
                           size=class_dim,
                           act=paddle.activation.Softmax())

    if is_infer:
        return prob
    else:
        cost = paddle.layer.classification_cost(input=prob, label=lbl)

        return cost, prob, lbl
