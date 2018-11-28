import paddle.fluid as fluid
import math

dense_feature_dim = 13


def ctr_dnn_model(embedding_size, sparse_feature_dim):
    dense_input = fluid.layers.data(
        name="dense_input", shape=[dense_feature_dim], dtype='float32')

    sparse_input_ids = [
        fluid.layers.data(
            name="C" + str(i), shape=[1], lod_level=1, dtype='int64')
        for i in range(1, 27)
    ]

    label = fluid.layers.data(name='label', shape=[1], dtype='int64')

    datas = [dense_input] + sparse_input_ids + [label]

    py_reader = fluid.layers.create_py_reader_by_data(capacity=64,
                                                      feed_list=datas,
                                                      name='py_reader',
                                                      use_double_buffer=True)
    words = fluid.layers.read_file(py_reader)

    def embedding_layer(input):
        return fluid.layers.embedding(
            input=input,
            is_sparse=True,
            # you need to patch https://github.com/PaddlePaddle/Paddle/pull/14190
            # if you want to set is_distributed to True
            is_distributed=False,
            size=[sparse_feature_dim, embedding_size],
            param_attr=fluid.ParamAttr(name="SparseFeatFactors", initializer=fluid.initializer.Uniform()))

    sparse_embed_seq = map(embedding_layer, words[1:-1])
    concated = fluid.layers.concat(sparse_embed_seq + words[0:1], axis=1)

    fc1 = fluid.layers.fc(input=concated, size=400, act='relu',
        param_attr=fluid.ParamAttr(initializer=fluid.initializer.Normal(scale=1/math.sqrt(concated.shape[1]))))
    fc2 = fluid.layers.fc(input=fc1, size=400, act='relu',
        param_attr=fluid.ParamAttr(initializer=fluid.initializer.Normal(scale=1/math.sqrt(fc1.shape[1]))))
    fc3 = fluid.layers.fc(input=fc2, size=400, act='relu',
        param_attr=fluid.ParamAttr(initializer=fluid.initializer.Normal(scale=1/math.sqrt(fc2.shape[1]))))
    predict = fluid.layers.fc(input=fc3, size=2, act='softmax',
        param_attr=fluid.ParamAttr(initializer=fluid.initializer.Normal(scale=1/math.sqrt(fc3.shape[1]))))

    cost = fluid.layers.cross_entropy(input=predict, label=words[-1:])
    avg_cost = fluid.layers.reduce_sum(cost)
    accuracy = fluid.layers.accuracy(input=predict, label=words[-1:])
    auc_var, batch_auc_var, auc_states = \
        fluid.layers.auc(input=predict, label=words[-1:], num_thresholds=2**12, slide_steps=20)

    return avg_cost, auc_var, batch_auc_var, py_reader
