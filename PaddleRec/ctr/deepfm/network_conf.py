import paddle.fluid as fluid
import math


def ctr_deepfm_model(embedding_size,
                     num_field,
                     num_feat,
                     layer_sizes,
                     act,
                     reg,
                     is_sparse=False):
    init_value_ = 0.1

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

    # -------------------- first order term  --------------------

    first_weights_re = fluid.embedding(
        input=feat_idx,
        is_sparse=is_sparse,
        dtype='float32',
        size=[num_feat + 1, 1],
        padding_idx=0,
        param_attr=fluid.ParamAttr(
            initializer=fluid.initializer.TruncatedNormalInitializer(
                loc=0.0, scale=init_value_),
            regularizer=fluid.regularizer.L1DecayRegularizer(reg)))
    first_weights = fluid.layers.reshape(
        first_weights_re, shape=[-1, num_field, 1])  # None * num_field * 1
    y_first_order = fluid.layers.reduce_sum((first_weights * feat_value), 1)

    # -------------------- second order term  --------------------

    feat_embeddings_re = fluid.embedding(
        input=feat_idx,
        is_sparse=is_sparse,
        dtype='float32',
        size=[num_feat + 1, embedding_size],
        padding_idx=0,
        param_attr=fluid.ParamAttr(
            initializer=fluid.initializer.TruncatedNormalInitializer(
                loc=0.0, scale=init_value_ / math.sqrt(float(embedding_size)))))
    feat_embeddings = fluid.layers.reshape(
        feat_embeddings_re,
        shape=[-1, num_field,
               embedding_size])  # None * num_field * embedding_size
    feat_embeddings = feat_embeddings * feat_value  # None * num_field * embedding_size

    # sum_square part
    summed_features_emb = fluid.layers.reduce_sum(feat_embeddings,
                                                  1)  # None * embedding_size
    summed_features_emb_square = fluid.layers.square(
        summed_features_emb)  # None * embedding_size

    # square_sum part
    squared_features_emb = fluid.layers.square(
        feat_embeddings)  # None * num_field * embedding_size
    squared_sum_features_emb = fluid.layers.reduce_sum(
        squared_features_emb, 1)  # None * embedding_size

    y_second_order = 0.5 * fluid.layers.reduce_sum(
        summed_features_emb_square - squared_sum_features_emb, 1,
        keep_dim=True)  # None * 1

    # -------------------- DNN --------------------

    y_dnn = fluid.layers.reshape(feat_embeddings,
                                 [-1, num_field * embedding_size])
    for s in layer_sizes:
        y_dnn = fluid.layers.fc(
            input=y_dnn,
            size=s,
            act=act,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.TruncatedNormalInitializer(
                    loc=0.0, scale=init_value_ / math.sqrt(float(10)))),
            bias_attr=fluid.ParamAttr(
                initializer=fluid.initializer.TruncatedNormalInitializer(
                    loc=0.0, scale=init_value_)))
    y_dnn = fluid.layers.fc(
        input=y_dnn,
        size=1,
        act=None,
        param_attr=fluid.ParamAttr(
            initializer=fluid.initializer.TruncatedNormalInitializer(
                loc=0.0, scale=init_value_)),
        bias_attr=fluid.ParamAttr(
            initializer=fluid.initializer.TruncatedNormalInitializer(
                loc=0.0, scale=init_value_)))

    # ------------------- DeepFM ------------------

    predict = fluid.layers.sigmoid(y_first_order + y_second_order + y_dnn)
    cost = fluid.layers.log_loss(input=predict, label=label)
    batch_cost = fluid.layers.reduce_sum(cost)

    # for auc
    predict_2d = fluid.layers.concat([1 - predict, predict], 1)
    label_int = fluid.layers.cast(label, 'int64')
    auc_var, batch_auc_var, auc_states = fluid.layers.auc(input=predict_2d,
                                                          label=label_int,
                                                          slide_steps=0)

    return batch_cost, auc_var, [raw_feat_idx, raw_feat_value,
                                 label], auc_states
