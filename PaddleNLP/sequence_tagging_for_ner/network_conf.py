import math

import paddle.fluid as fluid
from paddle.fluid.initializer import NormalInitializer

from utils import logger, load_dict, get_embedding


def ner_net(word_dict_len, label_dict_len, parallel, stack_num=2):
    mark_dict_len = 2
    word_dim = 50
    mark_dim = 5
    hidden_dim = 300
    IS_SPARSE = True
    embedding_name = 'emb'

    def _net_conf(word, mark, target):
        word_embedding = fluid.layers.embedding(
            input=word,
            size=[word_dict_len, word_dim],
            dtype='float32',
            is_sparse=IS_SPARSE,
            param_attr=fluid.ParamAttr(
                name=embedding_name, trainable=False))

        mark_embedding = fluid.layers.embedding(
            input=mark,
            size=[mark_dict_len, mark_dim],
            dtype='float32',
            is_sparse=IS_SPARSE)

        word_caps_vector = fluid.layers.concat(
            input=[word_embedding, mark_embedding], axis=1)
        mix_hidden_lr = 1

        rnn_para_attr = fluid.ParamAttr(
            initializer=NormalInitializer(
                loc=0.0, scale=0.0),
            learning_rate=mix_hidden_lr)
        hidden_para_attr = fluid.ParamAttr(
            initializer=NormalInitializer(
                loc=0.0, scale=(1. / math.sqrt(hidden_dim) / 3)),
            learning_rate=mix_hidden_lr)

        hidden = fluid.layers.fc(
            input=word_caps_vector,
            name="__hidden00__",
            size=hidden_dim,
            act="tanh",
            bias_attr=fluid.ParamAttr(initializer=NormalInitializer(
                loc=0.0, scale=(1. / math.sqrt(hidden_dim) / 3))),
            param_attr=fluid.ParamAttr(initializer=NormalInitializer(
                loc=0.0, scale=(1. / math.sqrt(hidden_dim) / 3))))
        fea = []
        for direction in ["fwd", "bwd"]:
            for i in range(stack_num):
                if i != 0:
                    hidden = fluid.layers.fc(
                        name="__hidden%02d_%s__" % (i, direction),
                        size=hidden_dim,
                        act="stanh",
                        bias_attr=fluid.ParamAttr(initializer=NormalInitializer(
                            loc=0.0, scale=1.0)),
                        input=[hidden, rnn[0], rnn[1]],
                        param_attr=[
                            hidden_para_attr, rnn_para_attr, rnn_para_attr
                        ])
                rnn = fluid.layers.dynamic_lstm(
                    name="__rnn%02d_%s__" % (i, direction),
                    input=hidden,
                    size=hidden_dim,
                    candidate_activation='relu',
                    gate_activation='sigmoid',
                    cell_activation='sigmoid',
                    bias_attr=fluid.ParamAttr(initializer=NormalInitializer(
                        loc=0.0, scale=1.0)),
                    is_reverse=(i % 2) if direction == "fwd" else not i % 2,
                    param_attr=rnn_para_attr)
            fea += [hidden, rnn[0], rnn[1]]

        rnn_fea = fluid.layers.fc(
            size=hidden_dim,
            bias_attr=fluid.ParamAttr(initializer=NormalInitializer(
                loc=0.0, scale=(1. / math.sqrt(hidden_dim) / 3))),
            act="stanh",
            input=fea,
            param_attr=[hidden_para_attr, rnn_para_attr, rnn_para_attr] * 2)

        emission = fluid.layers.fc(
            size=label_dict_len,
            input=rnn_fea,
            param_attr=fluid.ParamAttr(initializer=NormalInitializer(
                loc=0.0, scale=(1. / math.sqrt(hidden_dim) / 3))))

        crf_cost = fluid.layers.linear_chain_crf(
            input=emission,
            label=target,
            param_attr=fluid.ParamAttr(
                name='crfw',
                initializer=NormalInitializer(
                    loc=0.0, scale=(1. / math.sqrt(hidden_dim) / 3)),
                learning_rate=mix_hidden_lr))
        avg_cost = fluid.layers.mean(x=crf_cost)
        return avg_cost, emission

    word = fluid.layers.data(name='word', shape=[1], dtype='int64', lod_level=1)
    mark = fluid.layers.data(name='mark', shape=[1], dtype='int64', lod_level=1)
    target = fluid.layers.data(
        name="target", shape=[1], dtype='int64', lod_level=1)

    if parallel:
        places = fluid.layers.device.get_places()
        pd = fluid.layers.ParallelDo(places)
        with pd.do():
            word_ = pd.read_input(word)
            mark_ = pd.read_input(mark)
            target_ = pd.read_input(target)
            avg_cost, emission_base = _net_conf(word_, mark_, target_)
            pd.write_output(avg_cost)
            pd.write_output(emission_base)
        avg_cost_list, emission = pd()
        avg_cost = fluid.layers.mean(x=avg_cost_list)
        emission.stop_gradient = True
    else:
        avg_cost, emission = _net_conf(word, mark, target)

    return avg_cost, emission, word, mark, target
