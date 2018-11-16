import paddle.fluid as fluid
import math


def skip_gram_word2vec(dict_size,
                       embedding_size,
                       max_code_length=None,
                       with_hierarchical=False,
                       with_nce=True,
                       non_leaf_num=None):
    input_word = fluid.layers.data(name="input_word", shape=[1], dtype='int64')

    emb = fluid.layers.embedding(
        input=input_word,
        size=[dict_size, embedding_size],
        param_attr=fluid.ParamAttr(initializer=fluid.initializer.Normal(
            scale=1 / math.sqrt(dict_size))))

    predict_word = fluid.layers.data(
        name='predict_word', shape=[1], dtype='int64')

    data_list = [input_word, predict_word]

    cost = None

    if (not with_hierarchical) and with_nce:
        w_param_name = "nce_w"
        fluid.default_main_program().global_block().create_parameter(
            shape=[dict_size, embedding_size],
            dtype='float32',
            name=w_param_name)

        b_param_name = "nce_b"
        fluid.default_main_program().global_block().create_parameter(
            shape=[dict_size, 1], dtype='float32', name=b_param_name)

        cost = fluid.layers.nce(input=emb,
                                label=predict_word,
                                num_total_classes=dict_size,
                                param_attr=fluid.ParamAttr(name=w_param_name),
                                bias_attr=fluid.ParamAttr(name=b_param_name),
                                num_neg_samples=5)
    else:
        ptable = None
        pcode = None
        if max_code_length != None and with_hierarchical:
            ptable = fluid.layers.data(
                name='ptable', shape=[max_code_length], dtype='int64')
            pcode = fluid.layers.data(
                name='pcode', shape=[max_code_length], dtype='int64')
            data_list.append(pcode)
            data_list.append(ptable)
        else:
            ptable = fluid.layers.data(name='ptable', shape=[1], dtype='int64')
            pcode = fluid.layers.data(name='pcode', shape=[1], dtype='int64')
            data_list.append(pcode)
            data_list.append(ptable)
        if non_leaf_num == None:
            non_leaf_num = dict_size

        cost = fluid.layers.hsigmoid(
            input=emb,
            label=predict_word,
            non_leaf_num=non_leaf_num,
            ptable=ptable,
            pcode=pcode,
            is_costum=True)

    avg_cost = fluid.layers.reduce_mean(cost)

    return avg_cost, data_list
