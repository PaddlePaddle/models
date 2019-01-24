import paddle.fluid as fluid
import math

dense_feature_dim = 13

user_dense_feature_dim = 13
item_dense_feature_dim = 13

## text cnn conf
WORD_SIZE = 100000
EMBED_SIZE = 64
CNN_DIM = 128
CNN_FILTER_SIZE = 5


def text_cnn(word):
    """
    """
    embed = fluid.layers.embedding(
        input=word,
        size=[WORD_SIZE, EMBED_SIZE],
        dtype='float32',
        param_attr=fluid.ParamAttr(
                        initializer=fluid.initializer.Normal(scale=1/math.sqrt(WORD_SIZE))),
        is_sparse=IS_SPARSE,
        is_distributed=False)
    cnn = fluid.nets.sequence_conv_pool(
         input = embed,
         num_filters = CNN_DIM,
         filter_size = CNN_FILTER_SIZE,
         param_attr=fluid.ParamAttr(
                         initializer=fluid.initializer.Normal(scale=1/math.sqrt(CNN_FILTER_SIZE * embed.shape[1]))),
         act='tanh',
         pool_type = "max")
    return cnn


def deepmf_ctr_model(embedding_size, sparse_feature_dim):

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

    user_dense_input = fluid.layers.data(
        name="dense_input", shape=[user_dense_feature_dim], dtype='float32')

    user_sparse_input_ids = [
        fluid.layers.data(name="USER" + str(i), shape=[1], lod_level=1, dtype='int64')
        for i in range(1, user_sparse_slot_num)]

    item_dense_input = fluid.layers.data(
        name="dense_input", shape=[item_dense_feature_dim], dtype='float32')

    item_sparse_input_ids = [
        fluid.layers.data(name="ITEM" + str(i), shape=[1], lod_level=1, dtype='int64')
        for i in range(1, item_sparse_slot_num)]



    label = fluid.layers.data(name='label', shape=[1], dtype='int64')

    datas = [user_dense_input] + [item_dense_input] + user_sparse_input_ids  + item_sparse_input_ids + [label]

    py_reader = fluid.layers.create_py_reader_by_data(capacity=64,
                                                      feed_list=datas,
                                                      name='py_reader',
                                                      use_double_buffer=True)
    words = fluid.layers.read_file(py_reader)

    user_sparse_embed_seq = list(map(embedding_layer, words[2: user_sparse_slot_num + 2]))
    item_sparse_embed_seq = list(map(embedding_layer, words[user_sparse_slot_num + 2: user_sparse_slot_num + item_sparse_slot_num + 2]))
    
    
    user_concated = fluid.layers.concat(user_sparse_embed_seq + words[0:1], axis=1)
    item_concated = fluid.layers.concat(item_sparse_embed_seq + words[1:2], axis=1)

    user_fc1 = fluid.layers.fc(input=user_concated, size=400, act='relu',
                          param_attr=fluid.ParamAttr(initializer=fluid.initializer.Normal(
                              scale=1 / math.sqrt(concated.shape[1]))))
    user_fc2 = fluid.layers.fc(input=fc1, size=128, act='relu',
                          param_attr=fluid.ParamAttr(initializer=fluid.initializer.Normal(
                              scale=1 / math.sqrt(fc1.shape[1]))))
    user_fc3 = fluid.layers.fc(input=fc2, size=64, act='tanh',
                          param_attr=fluid.ParamAttr(initializer=fluid.initializer.Normal(
                              scale=1 / math.sqrt(fc2.shape[1]))))

    item_fc1 = fluid.layers.fc(input=user_concated, size=400, act='relu',
                          param_attr=fluid.ParamAttr(initializer=fluid.initializer.Normal(
                              scale=1 / math.sqrt(concated.shape[1]))))
    item_fc2 = fluid.layers.fc(input=fc1, size=128, act='relu',
                          param_attr=fluid.ParamAttr(initializer=fluid.initializer.Normal(
                              scale=1 / math.sqrt(fc1.shape[1]))))
    item_fc3 = fluid.layers.fc(input=fc2, size=64, act='tanh',
                          param_attr=fluid.ParamAttr(initializer=fluid.initializer.Normal(
                              scale=1 / math.sqrt(fc2.shape[1]))))

    sim = fluid.layers.cos_sim(X = user_fc3, Y = item_fc3)

    predict = fluid.layers.fc(input=sim, size=2, act='softmax',
                              param_attr=fluid.ParamAttr(initializer=fluid.initializer.Normal(
                                  scale=1 / math.sqrt(fc3.shape[1]))))

    cost = fluid.layers.cross_entropy(input=predict, label=words[-1])
    avg_cost = fluid.layers.reduce_sum(cost)
    accuracy = fluid.layers.accuracy(input=predict, label=words[-1])
    auc_var, batch_auc_var, auc_states = \
        fluid.layers.auc(input=predict, label=words[-1], num_thresholds=2 ** 12, slide_steps=20)

    return avg_cost, auc_var, batch_auc_var, py_reader

