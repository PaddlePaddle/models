import paddle.fluid as fluid

dense_feature_dim = 13
sparse_feature_dim = 117568


def DeepFM(factor_size, infer=False):
    dense_input = fluid.layers.data(
        name="dense_input", shape=[dense_feature_dim], dtype='float32')
    sparse_input_ids = [
        fluid.layers.data(
            name="C" + str(i), shape=[1], lod_level=1, dtype='int64')
        for i in range(1, 27)
    ]

    def embedding_layer(input):
        return fluid.layers.embedding(
            input=input,
            size=[sparse_feature_dim, factor_size],
            param_attr=fluid.ParamAttr(name="SparseFeatFactors"))

    sparse_embed_seq = map(embedding_layer, sparse_input_ids)
    concated = fluid.layers.concat(sparse_embed_seq + [dense_input], axis=1)

    fc1 = fluid.layers.fc(input=concated, size=400, act='relu')
    fc2 = fluid.layers.fc(input=fc1, size=400, act='relu')
    fc3 = fluid.layers.fc(input=fc2, size=400, act='relu')
    predict = fluid.layers.fc(input=fc3, size=2, act='sigmoid')

    data_list = [dense_input] + sparse_input_ids

    if not infer:
        label = fluid.layers.data(name='label', shape=[1], dtype='int64')
        cost = fluid.layers.cross_entropy(input=predict, label=label)
        avg_cost = fluid.layers.reduce_sum(cost)
        accuracy = fluid.layers.accuracy(input=predict, label=label)
        auc_var, cur_auc_var, auc_states = fluid.layers.auc(input=predict, label=label, num_thresholds=2**12)

        data_list.append(label)
        return avg_cost, data_list
    else:
        return predict, data_list
