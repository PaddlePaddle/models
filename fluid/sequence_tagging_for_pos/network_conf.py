# Copyright(c) 2018 PaddlePaddle.  All rights reserved.
# Created on 2018
# Author:Lin_Bo
# Version 1.0
# filename: network_conf.py

import paddle
from paddle import fluid
import paddle.fluid as fluid


def window_net(data,
               label,
               dict_dim,
               class_num,
               emb_dim=32,
               linear_layer_size=64,
               window_size=5):
    """
    define the topology of the window network

    :param data: the input data
    :type input_dim: paddle.fluid.framework.Variable
    :param label: the input label
    :type label:  paddle.fluid.framework.Variable
    :param dict_dim: size of word dictionary
    :type dict_dim: int
    :params class_num: number of instance class
    :type class_num: int
    :params emb_dim: embedding vector dimension
    :type emb_dim: int
    :params linear_layer_size: size of hidden linear layer
    :type linear_layer_size: int
    """
    # lookup table, the embedding layer
    emb = fluid.layers.embedding(
        input=data,
        size=[dict_dim, emb_dim],
        param_attr='emb.w',
        is_sparse=True)

    # Linear Layer of global full connect
    # the filter_size equal to emb size
    emb_fc_layer = fluid.nets.sequence_conv_pool(
        input=emb,
        num_filters=linear_layer_size,
        filter_size=window_size,
        act=None,
        pool_type="sum")

    # Tanh Layer
    tanh_layer = fluid.layers.tanh(emb_fc_layer)

    # prediction
    prediction = fluid.layers.fc(input=[tanh_layer],
                                 size=class_num,
                                 act="softmax")

    # cost and batch average cost
    cost = fluid.layers.cross_entropy(input=prediction, label=label)
    avg_cost = fluid.layers.mean(x=cost)

    # batch accuracy
    acc = fluid.layers.accuracy(input=prediction, label=label)

    return avg_cost, acc, prediction


def sentence_net(data,
                 label,
                 dict_dim,
                 class_num,
                 emb_dim=32,
                 hid_dim=64,
                 kernel_width=3,
                 tanh_layer_size=64):
    """
    sentence network definition

    :param data: the input data
    :type data: paddle.fluid.framework.Variable
    :param label: the input label
    :type label:  paddle.fluid.framework.Variable
    :params class_num: number of instance class
    :type class_num: int
    :params emb_dim: embedding vector dimension
    :type emb_dim: int
    :params hid_dim: number of same size convolution kernels
    :type hid_dim: int
    :params kernel_width: size of convolution kernels
    :type kernel_width: int
    :params tanh_layer_size: the size of hidden tanh layer
    :type tanh_layer_size: int
    """
    # lookup table, the embedding layer
    emb = fluid.layers.embedding(input=data, size=[dict_dim, emb_dim])

    # convolution layers with max pooling
    conv_layer = fluid.nets.sequence_conv_pool(
        input=emb,
        num_filters=hid_dim,
        filter_size=kernel_width,
        act=None,
        pool_type="max")

    # Linear Layer and tanh Layer
    tanh_layer = fluid.layers.fc(input=[conv_layer],
                                 size=tanh_layer_size,
                                 act="tanh")

    # prediction
    prediction = fluid.layers.fc(input=[tanh_layer],
                                 size=class_num,
                                 act="softmax")

    # cost and batch average cost
    cost = fluid.layers.cross_entropy(input=prediction, label=label)
    avg_cost = fluid.layers.mean(x=cost)

    # batch accuracy
    acc = fluid.layers.accuracy(input=prediction, label=label)

    return avg_cost, acc, prediction
