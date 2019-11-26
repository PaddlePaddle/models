import paddle.fluid as fluid
import math


def ctr_xdeepfm_model(embedding_size,
                      num_field,
                      num_feat,
                      layer_sizes_dnn,
                      act,
                      reg,
                      layer_sizes_cin,
                      is_sparse=False):
    init_value_ = 0.1
    initer = fluid.initializer.TruncatedNormalInitializer(
        loc=0.0, scale=init_value_)

    raw_feat_idx = fluid.data(
        name='feat_idx', shape=[None, num_field], dtype='int64')
    raw_feat_value = fluid.data(
        name='feat_value', shape=[None, num_field], dtype='float32')
    label = fluid.data(
        name='label', shape=[None, 1], dtype='float32')  # None * 1
    feat_idx = fluid.layers.reshape(raw_feat_idx,
                                    [-1, 1])  # (None * num_field) * 1
    feat_value = fluid.layers.reshape(
        raw_feat_value, [-1, num_field, 1])  # None * num_field * 1

    feat_embeddings = fluid.embedding(
        input=feat_idx,
        is_sparse=is_sparse,
        dtype='float32',
        size=[num_feat + 1, embedding_size],
        padding_idx=0,
        param_attr=fluid.ParamAttr(initializer=initer))
    feat_embeddings = fluid.layers.reshape(
        feat_embeddings,
        [-1, num_field, embedding_size])  # None * num_field * embedding_size
    feat_embeddings = feat_embeddings * feat_value  # None * num_field * embedding_size

    # -------------------- linear  --------------------

    weights_linear = fluid.embedding(
        input=feat_idx,
        is_sparse=is_sparse,
        dtype='float32',
        size=[num_feat + 1, 1],
        padding_idx=0,
        param_attr=fluid.ParamAttr(initializer=initer))
    weights_linear = fluid.layers.reshape(
        weights_linear, [-1, num_field, 1])  # None * num_field * 1
    b_linear = fluid.layers.create_parameter(
        shape=[1],
        dtype='float32',
        default_initializer=fluid.initializer.ConstantInitializer(value=0))
    y_linear = fluid.layers.reduce_sum(
        (weights_linear * feat_value), 1) + b_linear

    # -------------------- CIN  --------------------

    Xs = [feat_embeddings]
    last_s = num_field
    for s in layer_sizes_cin:
        # calculate Z^(k+1) with X^k and X^0
        X_0 = fluid.layers.reshape(
            fluid.layers.transpose(Xs[0], [0, 2, 1]),
            [-1, embedding_size, num_field,
             1])  # None, embedding_size, num_field, 1
        X_k = fluid.layers.reshape(
            fluid.layers.transpose(Xs[-1], [0, 2, 1]),
            [-1, embedding_size, 1, last_s])  # None, embedding_size, 1, last_s
        Z_k_1 = fluid.layers.matmul(
            X_0, X_k)  # None, embedding_size, num_field, last_s

        # compresses Z^(k+1) to X^(k+1)
        Z_k_1 = fluid.layers.reshape(Z_k_1, [
            -1, embedding_size, last_s * num_field
        ])  # None, embedding_size, last_s*num_field
        Z_k_1 = fluid.layers.transpose(
            Z_k_1, [0, 2, 1])  # None, s*num_field, embedding_size
        Z_k_1 = fluid.layers.reshape(
            Z_k_1, [-1, last_s * num_field, 1, embedding_size]
        )  # None, last_s*num_field, 1, embedding_size  (None, channal_in, h, w) 
        X_k_1 = fluid.layers.conv2d(
            Z_k_1,
            num_filters=s,
            filter_size=(1, 1),
            act=None,
            bias_attr=False,
            param_attr=fluid.ParamAttr(
                initializer=initer))  # None, s, 1, embedding_size
        X_k_1 = fluid.layers.reshape(
            X_k_1, [-1, s, embedding_size])  # None, s, embedding_size

        Xs.append(X_k_1)
        last_s = s

    # sum pooling
    y_cin = fluid.layers.concat(Xs[1:],
                                1)  # None, (num_field++), embedding_size
    y_cin = fluid.layers.reduce_sum(y_cin, -1)  # None, (num_field++)
    y_cin = fluid.layers.fc(input=y_cin,
                            size=1,
                            act=None,
                            param_attr=fluid.ParamAttr(initializer=initer),
                            bias_attr=None)
    y_cin = fluid.layers.reduce_sum(y_cin, dim=-1, keep_dim=True)

    # -------------------- DNN --------------------

    y_dnn = fluid.layers.reshape(feat_embeddings,
                                 [-1, num_field * embedding_size])
    for s in layer_sizes_dnn:
        y_dnn = fluid.layers.fc(input=y_dnn,
                                size=s,
                                act=act,
                                param_attr=fluid.ParamAttr(initializer=initer),
                                bias_attr=None)
    y_dnn = fluid.layers.fc(input=y_dnn,
                            size=1,
                            act=None,
                            param_attr=fluid.ParamAttr(initializer=initer),
                            bias_attr=None)

    # ------------------- xDeepFM ------------------

    predict = fluid.layers.sigmoid(y_linear + y_cin + y_dnn)
    cost = fluid.layers.log_loss(input=predict, label=label, epsilon=0.0000001)
    batch_cost = fluid.layers.reduce_mean(cost)

    # for auc
    predict_2d = fluid.layers.concat([1 - predict, predict], 1)
    label_int = fluid.layers.cast(label, 'int64')
    auc_var, batch_auc_var, auc_states = fluid.layers.auc(input=predict_2d,
                                                          label=label_int,
                                                          slide_steps=0)

    return batch_cost, auc_var, [raw_feat_idx, raw_feat_value,
                                 label], auc_states
