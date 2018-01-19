import math
import paddle.v2 as paddle

import reader

__all__ = ["training_net", "inference_net", "feeding"]

feeding = {
    reader.Q_IDS_STR: reader.Q_IDS,
    reader.E_IDS_STR: reader.E_IDS,
    reader.QE_COMM_STR: reader.QE_COMM,
    reader.EE_COMM_STR: reader.EE_COMM,
    reader.LABELS_STR: reader.LABELS
}


def get_embedding(input, word_vec_dim, wordvecs):
    """
    Define word embedding
    
    :param input: layer input
    :type input: LayerOutput
    :param word_vec_dim: dimension of the word embeddings
    :type word_vec_dim: int
    :param wordvecs: word embedding matrix
    :type wordvecs: numpy array
    :return: embedding
    :rtype: LayerOutput
    """
    return paddle.layer.embedding(
        input=input,
        size=word_vec_dim,
        param_attr=paddle.attr.ParamAttr(
            name="wordvecs", is_static=True, initializer=lambda _: wordvecs))


def encoding_question(question, q_lstm_dim, latent_chain_dim, word_vec_dim,
                      drop_rate, wordvecs, default_init_std, default_l2_rate):
    """
    Define network for encoding question

    :param question: question token ids
    :type question: LayerOutput
    :param q_lstm_dim: dimension of the question LSTM
    :type q_lstm_dim: int
    :param latent_chain_dim: dimension of the attention layer
    :type latent_chain_dim: int
    :param word_vec_dim: dimension of the word embeddings
    :type word_vec_dim: int
    :param drop_rate: dropout rate
    :type drop_rate: float
    :param wordvecs: word embedding matrix
    :type wordvecs: numpy array
    :param default_init_std: default initial standard deviation
    :type default_init_std: float
    :param default_l2_rate: default l2 rate
    :type default_l2_rate: float
    :return: question encoding
    :rtype: LayerOutput
    """
    # word embedding
    emb = get_embedding(question, word_vec_dim, wordvecs)

    # question LSTM
    wx = paddle.layer.fc(act=paddle.activation.Linear(),
                         size=q_lstm_dim * 4,
                         input=emb,
                         param_attr=paddle.attr.ParamAttr(
                             name="_q_hidden1.w0",
                             initial_std=default_init_std,
                             l2_rate=default_l2_rate),
                         bias_attr=paddle.attr.ParamAttr(
                             name="_q_hidden1.wbias",
                             initial_std=0,
                             l2_rate=default_l2_rate))
    q_rnn = paddle.layer.lstmemory(
        input=wx,
        bias_attr=paddle.attr.ParamAttr(
            name="_q_rnn1.wbias", initial_std=0, l2_rate=default_l2_rate),
        param_attr=paddle.attr.ParamAttr(
            name="_q_rnn1.w0",
            initial_std=default_init_std,
            l2_rate=default_l2_rate))
    q_rnn = paddle.layer.dropout(q_rnn, drop_rate)

    # self attention
    fc = paddle.layer.fc(act=paddle.activation.Tanh(),
                         size=latent_chain_dim,
                         input=q_rnn,
                         param_attr=paddle.attr.ParamAttr(
                             name="_attention_layer1.w0",
                             initial_std=default_init_std,
                             l2_rate=default_l2_rate),
                         bias_attr=False)
    weight = paddle.layer.fc(size=1,
                             act=paddle.activation.SequenceSoftmax(),
                             input=fc,
                             param_attr=paddle.attr.ParamAttr(
                                 name="_attention_weight.w0",
                                 initial_std=default_init_std,
                                 l2_rate=default_l2_rate),
                             bias_attr=False)

    scaled_q_rnn = paddle.layer.scaling(input=q_rnn, weight=weight)

    q_encoding = paddle.layer.pooling(
        input=scaled_q_rnn, pooling_type=paddle.pooling.Sum())
    return q_encoding


def encoding_evidence(evidence, qe_comm, ee_comm, q_encoding, e_lstm_dim,
                      word_vec_dim, com_vec_dim, drop_rate, wordvecs,
                      default_init_std, default_l2_rate):
    """
    Define network for encoding evidence

    :param qe_comm: qe.ecomm features
    :type qe_comm: LayerOutput
    :param ee_comm: ee.ecomm features
    :type ee_comm: LayerOutput
    :param q_encoding: question encoding, a fixed-length vector
    :type q_encoding: LayerOutput
    :param e_lstm_dim: dimension of the evidence LSTMs
    :type e_lstm_dim: int
    :param word_vec_dim: dimension of the word embeddings
    :type word_vec_dim: int
    :param com_vec_dim: dimension of the qe.comm and ee.comm feature embeddings
    :type com_vec_dim: int
    :param drop_rate: dropout rate
    :type drop_rate: float
    :param wordvecs: word embedding matrix
    :type wordvecs: numpy array
    :param default_init_std: default initial standard deviation
    :type default_init_std: float
    :param default_l2_rate: default l2 rate
    :type default_l2_rate: float
    :return: evidence encoding
    :rtype: LayerOutput
    """

    def lstm(idx, reverse, inputs):
        """LSTM wrapper"""
        bias_attr = paddle.attr.ParamAttr(
            name="_e_hidden%d.wbias" % idx,
            initial_std=0,
            l2_rate=default_l2_rate)
        with paddle.layer.mixed(size=e_lstm_dim * 4, bias_attr=bias_attr) as wx:
            for i, input in enumerate(inputs):
                param_attr = paddle.attr.ParamAttr(
                    name="_e_hidden%d.w%d" % (idx, i),
                    initial_std=default_init_std,
                    l2_rate=default_l2_rate)
                wx += paddle.layer.full_matrix_projection(
                    input=input, param_attr=param_attr)

        e_rnn = paddle.layer.lstmemory(
            input=wx,
            reverse=reverse,
            bias_attr=paddle.attr.ParamAttr(
                name="_e_rnn%d.wbias" % idx,
                initial_std=0,
                l2_rate=default_l2_rate),
            param_attr=paddle.attr.ParamAttr(
                name="_e_rnn%d.w0" % idx,
                initial_std=default_init_std,
                l2_rate=default_l2_rate))
        e_rnn = paddle.layer.dropout(e_rnn, drop_rate)
        return e_rnn

    # share word embeddings with question
    emb = get_embedding(evidence, word_vec_dim, wordvecs)

    # copy q_encoding len(evidence) times
    q_encoding_expand = paddle.layer.expand(
        input=q_encoding, expand_as=evidence)

    # feature embeddings
    comm_initial_std = 1 / math.sqrt(64.0)
    qe_comm_emb = paddle.layer.embedding(
        input=qe_comm,
        size=com_vec_dim,
        param_attr=paddle.attr.ParamAttr(
            name="_cw_embedding.w0",
            initial_std=comm_initial_std,
            l2_rate=default_l2_rate))

    ee_comm_emb = paddle.layer.embedding(
        input=ee_comm,
        size=com_vec_dim,
        param_attr=paddle.attr.ParamAttr(
            name="_eecom_embedding.w0",
            initial_std=comm_initial_std,
            l2_rate=default_l2_rate))

    # evidence LSTMs
    first_layer_extra_inputs = [q_encoding_expand, qe_comm_emb, ee_comm_emb]
    e_rnn1 = lstm(1, False, [emb] + first_layer_extra_inputs)
    e_rnn2 = lstm(2, True, [e_rnn1])
    e_rnn3 = lstm(3, False, [e_rnn2, e_rnn1])  # with cross layer links

    return e_rnn3


def define_data(dict_dim, label_num):
    """
    Define data layers

    :param dict_dim: number of words in the vocabulary
    :type dict_dim: int
    :param label_num: label numbers, BIO:3, BIO2:4
    :type label_num: int
    :return: data layers
    :rtype: tuple of LayerOutput
    """
    question = paddle.layer.data(
        name=reader.Q_IDS_STR,
        type=paddle.data_type.integer_value_sequence(dict_dim))

    evidence = paddle.layer.data(
        name=reader.E_IDS_STR,
        type=paddle.data_type.integer_value_sequence(dict_dim))

    qe_comm = paddle.layer.data(
        name=reader.QE_COMM_STR,
        type=paddle.data_type.integer_value_sequence(2))

    ee_comm = paddle.layer.data(
        name=reader.EE_COMM_STR,
        type=paddle.data_type.integer_value_sequence(2))

    label = paddle.layer.data(
        name=reader.LABELS_STR,
        type=paddle.data_type.integer_value_sequence(label_num),
        layer_attr=paddle.attr.ExtraAttr(device=-1))

    return question, evidence, qe_comm, ee_comm, label


def define_common_network(conf):
    """
    Define common network

    :param conf: network conf
    :return: CRF features, golden labels
    :rtype: tuple
    """
    # define data layers
    question, evidence, qe_comm, ee_comm, label = \
            define_data(conf.dict_dim, conf.label_num)

    # encode question
    q_encoding = encoding_question(question, conf.q_lstm_dim,
                                   conf.latent_chain_dim, conf.word_vec_dim,
                                   conf.drop_rate, conf.wordvecs,
                                   conf.default_init_std, conf.default_l2_rate)

    # encode evidence
    e_encoding = encoding_evidence(
        evidence, qe_comm, ee_comm, q_encoding, conf.e_lstm_dim,
        conf.word_vec_dim, conf.com_vec_dim, conf.drop_rate, conf.wordvecs,
        conf.default_init_std, conf.default_l2_rate)

    # pre-compute CRF features
    crf_feats = paddle.layer.fc(act=paddle.activation.Linear(),
                                input=e_encoding,
                                size=conf.label_num,
                                param_attr=paddle.attr.ParamAttr(
                                    name="_output.w0",
                                    initial_std=conf.default_init_std,
                                    l2_rate=conf.default_l2_rate),
                                bias_attr=False)
    return crf_feats, label


def training_net(conf):
    """
    Define training network

    :param conf: network conf
    :return: CRF cost
    :rtype: LayerOutput
    """
    e_encoding, label = define_common_network(conf)
    crf = paddle.layer.crf(input=e_encoding,
                           label=label,
                           size=conf.label_num,
                           param_attr=paddle.attr.ParamAttr(
                               name="_crf.w0",
                               initial_std=conf.default_init_std,
                               l2_rate=conf.default_l2_rate),
                           layer_attr=paddle.attr.ExtraAttr(device=-1))

    return crf


def inference_net(conf):
    """
    Define training network

    :param conf: network conf
    :return: CRF viberbi decoding result
    :rtype: LayerOutput
    """
    e_encoding, label = define_common_network(conf)
    ret = paddle.layer.crf_decoding(
        input=e_encoding,
        size=conf.label_num,
        param_attr=paddle.attr.ParamAttr(name="_crf.w0"),
        layer_attr=paddle.attr.ExtraAttr(device=-1))

    return ret
