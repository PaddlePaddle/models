import paddle.fluid as fluid
import math


def skip_gram_word2vec(dict_size, embedding_size):
    input_word = fluid.layers.data(name="input_word", shape=[1], dtype='int64')

    emb = fluid.layers.embedding(
        input=input_word,
        size=[dict_size, embedding_size],
        param_attr=fluid.ParamAttr(initializer=fluid.initializer.Normal(
            scale=1 / math.sqrt(dict_size))))

    predict_word = fluid.layers.data(
        name='predict_word', shape=[1], dtype='int64')

    data_list = [input_word, predict_word]

    w_param_name = "nce_w"
    fluid.default_main_program().global_block().create_parameter(
        shape=[dict_size, embedding_size], dtype='float32', name=w_param_name)

    b_param_name = "nce_b"
    fluid.default_main_program().global_block().create_parameter(
        shape=[dict_size, 1], dtype='float32', name=b_param_name)

    cost = fluid.layers.nce(input=emb,
                            label=predict_word,
                            num_total_classes=dict_size,
                            param_attr=fluid.ParamAttr(name=w_param_name),
                            bias_attr=fluid.ParamAttr(name=b_param_name),
                            num_neg_samples=5)
    avg_cost = fluid.layers.reduce_mean(cost)

    return avg_cost, data_list
