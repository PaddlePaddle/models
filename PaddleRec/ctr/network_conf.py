import paddle.fluid as fluid
import math

dense_feature_dim = 13


def ctr_deepfm_model(factor_size, sparse_feature_dim, dense_feature_dim, sparse_input):
    def dense_fm_layer(input, emb_dict_size, factor_size, fm_param_attr):
        """
        dense_fm_layer
        """
        first_order = fluid.layers.fc(input=input, size=1)
        emb_table = fluid.layers.create_parameter(shape=[emb_dict_size, factor_size],
                                                  dtype='float32', attr=fm_param_attr)

        input_mul_factor = fluid.layers.matmul(input, emb_table)
        input_mul_factor_square = fluid.layers.square(input_mul_factor)
        input_square = fluid.layers.square(input)
        factor_square = fluid.layers.square(emb_table)
        input_square_mul_factor_square = fluid.layers.matmul(input_square, factor_square)

        second_order = 0.5 * (input_mul_factor_square - input_square_mul_factor_square)
        return first_order, second_order

    def sparse_fm_layer(input, emb_dict_size, factor_size, fm_param_attr):
        """
        sparse_fm_layer
        """
        first_embeddings = fluid.layers.embedding(
            input=input, dtype='float32', size=[emb_dict_size, 1], is_sparse=True)
        first_order = fluid.layers.sequence_pool(input=first_embeddings, pool_type='sum')

        nonzero_embeddings = fluid.layers.embedding(
            input=input, dtype='float32', size=[emb_dict_size, factor_size],
            param_attr=fm_param_attr, is_sparse=True)
        summed_features_emb = fluid.layers.sequence_pool(input=nonzero_embeddings, pool_type='sum')
        summed_features_emb_square = fluid.layers.square(summed_features_emb)

        squared_features_emb = fluid.layers.square(nonzero_embeddings)
        squared_sum_features_emb = fluid.layers.sequence_pool(
            input=squared_features_emb, pool_type='sum')

        second_order = 0.5 * (summed_features_emb_square - squared_sum_features_emb)
        return first_order, second_order

    dense_input = fluid.layers.data(name="dense_input", shape=[dense_feature_dim], dtype='float32')

    sparse_input_ids = [
        fluid.layers.data(name="C" + str(i), shape=[1], lod_level=1, dtype='int64')
        for i in range(1, 27)]

    label = fluid.layers.data(name='label', shape=[1], dtype='int64')

    datas = [dense_input] + sparse_input_ids + [label]

    py_reader = fluid.layers.create_py_reader_by_data(capacity=64,
                                                      feed_list=datas,
                                                      name='py_reader',
                                                      use_double_buffer=True)
    words = fluid.layers.read_file(py_reader)

    sparse_fm_param_attr = fluid.param_attr.ParamAttr(name="SparseFeatFactors",
                                                      initializer=fluid.initializer.Normal(
                                                          scale=1 / math.sqrt(sparse_feature_dim)))
    dense_fm_param_attr = fluid.param_attr.ParamAttr(name="DenseFeatFactors",
                                                     initializer=fluid.initializer.Normal(
                                                         scale=1 / math.sqrt(dense_feature_dim)))

    sparse_fm_first, sparse_fm_second = sparse_fm_layer(
        sparse_input, sparse_feature_dim, factor_size, sparse_fm_param_attr)
    dense_fm_first, dense_fm_second = dense_fm_layer(
        dense_input, dense_feature_dim, factor_size, dense_fm_param_attr)

    def embedding_layer(input):
        """embedding_layer"""
        emb = fluid.layers.embedding(
            input=input, dtype='float32', size=[sparse_feature_dim, factor_size],
            param_attr=sparse_fm_param_attr, is_sparse=True)
        return fluid.layers.sequence_pool(input=emb, pool_type='average')

    sparse_embed_seq = list(map(embedding_layer, sparse_input_ids))
    concated = fluid.layers.concat(sparse_embed_seq + [dense_input], axis=1)
    fc1 = fluid.layers.fc(input=concated, size=400, act='relu',
                          param_attr=fluid.ParamAttr(initializer=fluid.initializer.Normal(
                              scale=1 / math.sqrt(concated.shape[1]))))
    fc2 = fluid.layers.fc(input=fc1, size=400, act='relu',
                          param_attr=fluid.ParamAttr(initializer=fluid.initializer.Normal(
                              scale=1 / math.sqrt(fc1.shape[1]))))
    fc3 = fluid.layers.fc(input=fc2, size=400, act='relu',
                          param_attr=fluid.ParamAttr(initializer=fluid.initializer.Normal(
                              scale=1 / math.sqrt(fc2.shape[1]))))
    predict = fluid.layers.fc(
        input=[sparse_fm_first, sparse_fm_second, dense_fm_first, dense_fm_second, fc3],
        size=2,
        act="softmax",
        param_attr=fluid.ParamAttr(initializer=fluid.initializer.Normal(scale=1 / math.sqrt(fc3.shape[1]))))

    cost = fluid.layers.cross_entropy(input=predict, label=words[-1])
    avg_cost = fluid.layers.reduce_sum(cost)
    accuracy = fluid.layers.accuracy(input=predict, label=words[-1])
    auc_var, batch_auc_var, auc_states = \
        fluid.layers.auc(input=predict, label=words[-1], num_thresholds=2 ** 12, slide_steps=20)

    return avg_cost, auc_var, batch_auc_var, py_reader


def ctr_dnn_model(embedding_size, sparse_feature_dim, use_py_reader=True):

    def embedding_layer(input):
        return fluid.layers.embedding(
            input=input,
            is_sparse=True,
            # you need to patch https://github.com/PaddlePaddle/Paddle/pull/14190
            # if you want to set is_distributed to True
            is_distributed=False,
            size=[sparse_feature_dim, embedding_size],
            param_attr=fluid.ParamAttr(name="SparseFeatFactors",
                                       initializer=fluid.initializer.Uniform()))

    dense_input = fluid.layers.data(
        name="dense_input", shape=[dense_feature_dim], dtype='float32')

    sparse_input_ids = [
        fluid.layers.data(name="C" + str(i), shape=[1], lod_level=1, dtype='int64')
        for i in range(1, 27)]

    label = fluid.layers.data(name='label', shape=[1], dtype='int64')

    words = [dense_input] + sparse_input_ids + [label]

    py_reader = None
    if use_py_reader:
        py_reader = fluid.layers.create_py_reader_by_data(capacity=64,
                                                          feed_list=words,
                                                          name='py_reader',
                                                          use_double_buffer=True)
        words = fluid.layers.read_file(py_reader)

    sparse_embed_seq = list(map(embedding_layer, words[1:-1]))
    concated = fluid.layers.concat(sparse_embed_seq + words[0:1], axis=1)

    fc1 = fluid.layers.fc(input=concated, size=400, act='relu',
                          param_attr=fluid.ParamAttr(initializer=fluid.initializer.Normal(
                              scale=1 / math.sqrt(concated.shape[1]))))
    fc2 = fluid.layers.fc(input=fc1, size=400, act='relu',
                          param_attr=fluid.ParamAttr(initializer=fluid.initializer.Normal(
                              scale=1 / math.sqrt(fc1.shape[1]))))
    fc3 = fluid.layers.fc(input=fc2, size=400, act='relu',
                          param_attr=fluid.ParamAttr(initializer=fluid.initializer.Normal(
                              scale=1 / math.sqrt(fc2.shape[1]))))
    predict = fluid.layers.fc(input=fc3, size=2, act='softmax',
                              param_attr=fluid.ParamAttr(initializer=fluid.initializer.Normal(
                                  scale=1 / math.sqrt(fc3.shape[1]))))

    cost = fluid.layers.cross_entropy(input=predict, label=words[-1])
    avg_cost = fluid.layers.reduce_sum(cost)
    accuracy = fluid.layers.accuracy(input=predict, label=words[-1])
    auc_var, batch_auc_var, auc_states = \
        fluid.layers.auc(input=predict, label=words[-1], num_thresholds=2 ** 12, slide_steps=20)

    return avg_cost, auc_var, batch_auc_var, py_reader, words
