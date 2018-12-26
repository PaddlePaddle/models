import paddle.fluid as fluid

def network(vocab_size,
        hid_size=100,
        init_low_bound=-0.04,
        init_high_bound=0.04):
    """ network definition """
    emb_lr_x = 10.0
    gru_lr_x = 1.0
    fc_lr_x = 1.0
    # Input data
    src_wordseq = fluid.layers.data(
        name="src_wordseq", shape=[1], dtype="int64", lod_level=1)
    dst_wordseq = fluid.layers.data(
        name="dst_wordseq", shape=[1], dtype="int64", lod_level=1)

    emb = fluid.layers.embedding(
        input=src_wordseq,
        size=[vocab_size, hid_size],
        param_attr=fluid.ParamAttr(
            initializer=fluid.initializer.Uniform(
                low=init_low_bound, high=init_high_bound),
            learning_rate=emb_lr_x),
        is_sparse=True)
    fc0 = fluid.layers.fc(input=emb,
                          size=hid_size * 3,
                          param_attr=fluid.ParamAttr(
                              initializer=fluid.initializer.Uniform(
                                  low=init_low_bound, high=init_high_bound),
                              learning_rate=gru_lr_x))
    gru_h0 = fluid.layers.dynamic_gru(
        input=fc0,
        size=hid_size,
        param_attr=fluid.ParamAttr(
            initializer=fluid.initializer.Uniform(
                low=init_low_bound, high=init_high_bound),
            learning_rate=gru_lr_x))

    fc = fluid.layers.fc(input=gru_h0,
                         size=vocab_size,
                         act='softmax',
                         param_attr=fluid.ParamAttr(
                             initializer=fluid.initializer.Uniform(
                                 low=init_low_bound, high=init_high_bound),
                             learning_rate=fc_lr_x))

    cost = fluid.layers.cross_entropy(input=fc, label=dst_wordseq)
    acc = fluid.layers.accuracy(input=fc, label=dst_wordseq, k=20)
    avg_cost = fluid.layers.mean(x=cost)
    return src_wordseq, dst_wordseq, avg_cost, acc
