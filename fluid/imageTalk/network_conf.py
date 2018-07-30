# Copyright(c) 2018 PaddlePaddle.  All rights reserved.
# Created on 2018
# Author:Lin_Bo
# Version 1.0
# filename: network_conf.py

import paddle.fluid as fluid


def lstm_net(prev_hidden,
             prev_cell,
             word_dict_dim,
             pre_word,
             lstm_him_dim=128,
             emb_dim=128):

    S_t = fluid.layers.embedding(
        input=[pre_word], size=[word_dict_dim, emb_dim])
    x_t = fluid.layers.fc(input=[S_t], size=lstm_him_dim)

    hidden, cell = lstm_cell(
        x_t=x_t, hidden_t_prev=prev_hidden, cell_t_prev=prev_cell)

    hidden_out = fluid.layers.fc(input=[hidden], size=4096)
    cell_out = fluid.layers.fc(input=[cell], size=4096)

    return hidden_out, cell_out


def lstm_prediction(lstm_hidden, word_dict_dim, word):
    # prediction
    prediction = fluid.layers.fc(input=[lstm_hidden],
                                 size=word_dict_dim,
                                 act="softmax")

    cost = fluid.layers.cross_entropy(input=prediction, label=word)

    # batch accuracy
    acc = fluid.layers.accuracy(input=prediction, label=word)

    avg_cost = fluid.layers.mean(x=cost)

    return avg_cost, acc, prediction


def lstm_cell(x_t, hidden_t_prev, cell_t_prev, lstm_him_dim=128):

    ## input gate
    i_tx = fluid.layers.fc(input=[x_t], size=lstm_him_dim, act=None)
    i_th = fluid.layers.fc(input=[hidden_t_prev], size=lstm_him_dim, act=None)
    i_t = fluid.layers.sums([i_tx, i_th])
    i_t = fluid.layers.sigmoid(i_t)

    ## forget gate
    f_tx = fluid.layers.fc(input=[x_t], size=lstm_him_dim, act=None)
    f_th = fluid.layers.fc(input=[hidden_t_prev], size=lstm_him_dim, act=None)
    f_t = fluid.layers.sums([f_tx, f_th])
    f_t = fluid.layers.sigmoid(f_t)

    ## lstm cell
    c_tx = fluid.layers.fc(input=[x_t], size=lstm_him_dim, act=None)
    c_th = fluid.layers.fc(input=[hidden_t_prev], size=lstm_him_dim, act=None)
    c_t_tanh = fluid.layers.tanh(fluid.layers.sums([c_tx, c_th]))

    c_tf = fluid.layers.elementwise_mul(cell_t_prev, f_t)
    c_ti = fluid.layers.elementwise_mul(c_t_tanh, i_t)

    c_t = fluid.layers.sums([c_tf, c_ti])
    c_t = fluid.layers.tanh(c_t)

    ## output gate
    o_tx = fluid.layers.fc(input=[x_t], size=lstm_him_dim, act=None)
    o_th = fluid.layers.fc(input=[hidden_t_prev], size=lstm_him_dim, act=None)
    o_t = fluid.layers.sums([o_tx, o_th])
    o_t = fluid.layers.sigmoid(o_t)

    ## output hidden
    h_t = fluid.layers.elementwise_mul(c_t, o_t)

    return h_t, c_t


def lstm_main(prev_hidden,
              prev_cell,
              word_dict_dim,
              lstm_him_dim=128,
              emb_dim=128,
              infer_dim=4096,
              pre_word=None,
              word=None):
    """
    define the topology of the lstm network

    :param prev_hidden: the lstm input data
    :type prev_hidden: paddle.fluid.framework.Variable
    :param prev_cell: the input lstm cell data
    :type prev_cell:  paddle.fluid.framework.Variable
    :param word_dict_dim: size of word dictionary
    :type word_dict_dim: int
    :params infer_dim: the input and output infer data dimension
    :type infer_dim: int
    :params emb_dim: embedding vector dimension
    :type emb_dim: int
    :params lstm_him_dim: size of lstm hidden linear layer
    :type lstm_him_dim: int
    :params pre_word: the result at the time of t-1 for input at the time of t
    :type pre_word: int
    :params word: the label at the time of t
    :type word: int
    """
    prev_hidden = fluid.layers.fc(input=[prev_hidden], size=lstm_him_dim)
    prev_cell = fluid.layers.fc(input=[prev_cell], size=lstm_him_dim)

    prev_hidden, prev_cell = lstm_net(
        prev_hidden=prev_hidden,
        prev_cell=prev_cell,
        word_dict_dim=word_dict_dim,
        pre_word=pre_word,
        lstm_him_dim=lstm_him_dim,
        emb_dim=emb_dim)

    avg_cost, acc, prediction = lstm_prediction(prev_hidden, word_dict_dim,
                                                word)

    prev_hidden = fluid.layers.fc(input=[prev_hidden], size=infer_dim)
    prev_cell = fluid.layers.fc(input=[prev_cell], size=infer_dim)

    return avg_cost, acc, prediction, prev_hidden, prev_cell
