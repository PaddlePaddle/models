import paddle.fluid as fluid


def all_vocab_network(vocab_size,
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


def train_bpr_network(vocab_size, neg_size, hid_size, drop_out=0.2):
    """ network definition """
    emb_lr_x = 1.0
    gru_lr_x = 1.0
    fc_lr_x = 1.0
    # Input data
    src = fluid.layers.data(name="src", shape=[1], dtype="int64", lod_level=1)
    pos_label = fluid.layers.data(
        name="pos_label", shape=[1], dtype="int64", lod_level=1)
    label = fluid.layers.data(
        name="label", shape=[neg_size + 1], dtype="int64", lod_level=1)

    emb_src = fluid.layers.embedding(
        input=src,
        size=[vocab_size, hid_size],
        param_attr=fluid.ParamAttr(
            name="emb",
            initializer=fluid.initializer.XavierInitializer(),
            learning_rate=emb_lr_x))

    emb_src_drop = fluid.layers.dropout(emb_src, dropout_prob=drop_out)

    fc0 = fluid.layers.fc(input=emb_src_drop,
                          size=hid_size * 3,
                          param_attr=fluid.ParamAttr(
                              name="gru_fc",
                              initializer=fluid.initializer.XavierInitializer(),
                              learning_rate=gru_lr_x),
                          bias_attr=False)
    gru_h0 = fluid.layers.dynamic_gru(
        input=fc0,
        size=hid_size,
        param_attr=fluid.ParamAttr(
            name="dy_gru.param",
            initializer=fluid.initializer.XavierInitializer(),
            learning_rate=gru_lr_x),
        bias_attr="dy_gru.bias")
    gru_h0_drop = fluid.layers.dropout(gru_h0, dropout_prob=drop_out)

    label_re = fluid.layers.sequence_reshape(input=label, new_dim=1)
    emb_label = fluid.layers.embedding(
        input=label_re,
        size=[vocab_size, hid_size],
        param_attr=fluid.ParamAttr(
            name="emb",
            initializer=fluid.initializer.XavierInitializer(),
            learning_rate=emb_lr_x))

    emb_label_drop = fluid.layers.dropout(emb_label, dropout_prob=drop_out)

    gru_exp = fluid.layers.expand(
        x=gru_h0_drop, expand_times=[1, (neg_size + 1)])
    gru = fluid.layers.sequence_reshape(input=gru_exp, new_dim=hid_size)

    ele_mul = fluid.layers.elementwise_mul(emb_label_drop, gru)
    red_sum = fluid.layers.reduce_sum(input=ele_mul, dim=1, keep_dim=True)

    pre = fluid.layers.sequence_reshape(input=red_sum, new_dim=(neg_size + 1))

    cost = fluid.layers.bpr_loss(input=pre, label=pos_label)
    cost_sum = fluid.layers.reduce_sum(input=cost)
    return src, pos_label, label, cost_sum


def train_cross_entropy_network(vocab_size, neg_size, hid_size, drop_out=0.2):
    """ network definition """
    emb_lr_x = 1.0
    gru_lr_x = 1.0
    fc_lr_x = 1.0
    # Input data
    src = fluid.layers.data(name="src", shape=[1], dtype="int64", lod_level=1)
    pos_label = fluid.layers.data(
        name="pos_label", shape=[1], dtype="int64", lod_level=1)
    label = fluid.layers.data(
        name="label", shape=[neg_size + 1], dtype="int64", lod_level=1)

    emb_src = fluid.layers.embedding(
        input=src,
        size=[vocab_size, hid_size],
        param_attr=fluid.ParamAttr(
            name="emb",
            initializer=fluid.initializer.XavierInitializer(),
            learning_rate=emb_lr_x))

    emb_src_drop = fluid.layers.dropout(emb_src, dropout_prob=drop_out)

    fc0 = fluid.layers.fc(input=emb_src_drop,
                          size=hid_size * 3,
                          param_attr=fluid.ParamAttr(
                              name="gru_fc",
                              initializer=fluid.initializer.XavierInitializer(),
                              learning_rate=gru_lr_x),
                          bias_attr=False)
    gru_h0 = fluid.layers.dynamic_gru(
        input=fc0,
        size=hid_size,
        param_attr=fluid.ParamAttr(
            name="dy_gru.param",
            initializer=fluid.initializer.XavierInitializer(),
            learning_rate=gru_lr_x),
        bias_attr="dy_gru.bias")
    gru_h0_drop = fluid.layers.dropout(gru_h0, dropout_prob=drop_out)

    label_re = fluid.layers.sequence_reshape(input=label, new_dim=1)
    emb_label = fluid.layers.embedding(
        input=label_re,
        size=[vocab_size, hid_size],
        param_attr=fluid.ParamAttr(
            name="emb",
            initializer=fluid.initializer.XavierInitializer(),
            learning_rate=emb_lr_x))

    emb_label_drop = fluid.layers.dropout(emb_label, dropout_prob=drop_out)

    gru_exp = fluid.layers.expand(
        x=gru_h0_drop, expand_times=[1, (neg_size + 1)])
    gru = fluid.layers.sequence_reshape(input=gru_exp, new_dim=hid_size)

    ele_mul = fluid.layers.elementwise_mul(emb_label_drop, gru)
    red_sum = fluid.layers.reduce_sum(input=ele_mul, dim=1, keep_dim=True)

    pre_ = fluid.layers.sequence_reshape(input=red_sum, new_dim=(neg_size + 1))
    pre = fluid.layers.softmax(input=pre_)

    cost = fluid.layers.cross_entropy(input=pre, label=pos_label)
    cost_sum = fluid.layers.reduce_sum(input=cost)
    return src, pos_label, label, cost_sum


def infer_network(vocab_size, batch_size, hid_size, dropout=0.2):
    src = fluid.layers.data(name="src", shape=[1], dtype="int64", lod_level=1)
    emb_src = fluid.layers.embedding(
        input=src, size=[vocab_size, hid_size], param_attr="emb")
    emb_src_drop = fluid.layers.dropout(
        emb_src, dropout_prob=dropout, is_test=True)

    fc0 = fluid.layers.fc(input=emb_src_drop,
                          size=hid_size * 3,
                          param_attr="gru_fc",
                          bias_attr=False)
    gru_h0 = fluid.layers.dynamic_gru(
        input=fc0,
        size=hid_size,
        param_attr="dy_gru.param",
        bias_attr="dy_gru.bias")
    gru_h0_drop = fluid.layers.dropout(
        gru_h0, dropout_prob=dropout, is_test=True)

    all_label = fluid.layers.data(
        name="all_label",
        shape=[vocab_size, 1],
        dtype="int64",
        append_batch_size=False)
    emb_all_label = fluid.layers.embedding(
        input=all_label, size=[vocab_size, hid_size], param_attr="emb")
    emb_all_label_drop = fluid.layers.dropout(
        emb_all_label, dropout_prob=dropout, is_test=True)

    all_pre = fluid.layers.matmul(
        gru_h0_drop, emb_all_label_drop, transpose_y=True)

    pos_label = fluid.layers.data(
        name="pos_label", shape=[1], dtype="int64", lod_level=1)
    acc = fluid.layers.accuracy(input=all_pre, label=pos_label, k=20)
    return acc
