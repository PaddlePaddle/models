import paddle.v2 as paddle


def cnn_cov_group(group_input, hidden_size):
    conv3 = paddle.networks.sequence_conv_pool(
        input=group_input, context_len=3, hidden_size=hidden_size)
    conv4 = paddle.networks.sequence_conv_pool(
        input=group_input, context_len=4, hidden_size=hidden_size)
    output_group = paddle.layer.fc(
        input=[conv3, conv4],
        size=hidden_size,
        param_attr=paddle.attr.ParamAttr(name='_cov_value_weight'),
        bias_attr=paddle.attr.ParamAttr(name='_cov_value_bias'),
        act=paddle.activation.Linear())
    return output_group


def nest_net(dict_dim,
             emb_size=28,
             hidden_size=128,
             class_num=2,
             is_infer=False):

    data = paddle.layer.data(
        "word", paddle.data_type.integer_value_sub_sequence(dict_dim))

    emb = paddle.layer.embedding(input=data, size=emb_size)
    nest_group = paddle.layer.recurrent_group(
        input=[paddle.layer.SubsequenceInput(emb), hidden_size],
        step=cnn_cov_group)
    avg_pool = paddle.layer.pooling(
        input=nest_group,
        pooling_type=paddle.pooling.Avg(),
        agg_level=paddle.layer.AggregateLevel.TO_NO_SEQUENCE)
    prob = paddle.layer.mixed(
        size=class_num,
        input=[paddle.layer.full_matrix_projection(input=avg_pool)],
        act=paddle.activation.Softmax())
    if is_infer == False:
        label = paddle.layer.data("label",
                                  paddle.data_type.integer_value(class_num))
        cost = paddle.layer.classification_cost(input=prob, label=label)
        return cost, prob, label

    return prob
