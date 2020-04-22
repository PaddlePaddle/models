import paddle.fluid as fluid


def cnn_net(dict_dim=100,
            max_len=10,
            cnn_dim=32,
            cnn_filter_size=128,
            emb_dim=8,
            hid_dim=128,
            class_dim=2,
            is_prediction=False):
    """
    Conv net
    """
    data = fluid.data(name="input", shape=[None, max_len], dtype='int64')
    label = fluid.data(name="label", shape=[None, 1], dtype='int64')
    seq_len = fluid.data(name="seq_len", shape=[None], dtype='int64')
    # embedding layer
    emb = fluid.embedding(input=data, size=[dict_dim, emb_dim])
    emb = fluid.layers.sequence_unpad(emb, length=seq_len)
    # convolution layer
    conv = fluid.nets.sequence_conv_pool(
        input=emb,
        num_filters=cnn_dim,
        filter_size=cnn_filter_size,
        act="tanh",
        pool_type="max")

    # full connect layer
    fc_1 = fluid.layers.fc(input=[conv], size=hid_dim)
    # softmax layer
    prediction = fluid.layers.fc(input=[fc_1], size=class_dim, act="softmax")
    #if is_prediction:
    #    return prediction
    cost = fluid.layers.cross_entropy(input=prediction, label=label)
    avg_cost = fluid.layers.mean(x=cost)
    acc = fluid.layers.accuracy(input=prediction, label=label)

    return avg_cost
