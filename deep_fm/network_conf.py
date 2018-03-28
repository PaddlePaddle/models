import paddle.v2 as paddle

dense_feature_dim = 13
sparse_feature_dim = 117568


def fm_layer(input, factor_size, fm_param_attr):
    first_order = paddle.layer.fc(input=input,
                                  size=1,
                                  act=paddle.activation.Linear())
    second_order = paddle.layer.factorization_machine(
        input=input,
        factor_size=factor_size,
        act=paddle.activation.Linear(),
        param_attr=fm_param_attr)
    out = paddle.layer.addto(
        input=[first_order, second_order],
        act=paddle.activation.Linear(),
        bias_attr=False)
    return out


def DeepFM(factor_size, infer=False):
    dense_input = paddle.layer.data(
        name="dense_input",
        type=paddle.data_type.dense_vector(dense_feature_dim))
    sparse_input = paddle.layer.data(
        name="sparse_input",
        type=paddle.data_type.sparse_binary_vector(sparse_feature_dim))
    sparse_input_ids = [
        paddle.layer.data(
            name="C" + str(i),
            type=paddle.data_type.integer_value(sparse_feature_dim))
        for i in range(1, 27)
    ]

    dense_fm = fm_layer(
        dense_input,
        factor_size,
        fm_param_attr=paddle.attr.Param(name="DenseFeatFactors"))
    sparse_fm = fm_layer(
        sparse_input,
        factor_size,
        fm_param_attr=paddle.attr.Param(name="SparseFeatFactors"))

    def embedding_layer(input):
        return paddle.layer.embedding(
            input=input,
            size=factor_size,
            param_attr=paddle.attr.Param(name="SparseFeatFactors"))

    sparse_embed_seq = map(embedding_layer, sparse_input_ids)
    sparse_embed = paddle.layer.concat(sparse_embed_seq)

    fc1 = paddle.layer.fc(input=[sparse_embed, dense_input],
                          size=400,
                          act=paddle.activation.Relu())
    fc2 = paddle.layer.fc(input=fc1, size=400, act=paddle.activation.Relu())
    fc3 = paddle.layer.fc(input=fc2, size=400, act=paddle.activation.Relu())

    predict = paddle.layer.fc(input=[dense_fm, sparse_fm, fc3],
                              size=1,
                              act=paddle.activation.Sigmoid())

    if not infer:
        label = paddle.layer.data(
            name="label", type=paddle.data_type.dense_vector(1))
        cost = paddle.layer.multi_binary_label_cross_entropy_cost(
            input=predict, label=label)
        paddle.evaluator.classification_error(
            name="classification_error", input=predict, label=label)
        paddle.evaluator.auc(name="auc", input=predict, label=label)
        return cost
    else:
        return predict
