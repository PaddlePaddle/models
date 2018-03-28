import paddle.v2 as paddle
from config import ModelConfig as conf


def cnn_cov_group(group_input, hidden_size):
    """
    Convolution group definition.
    :param group_input: The input of this layer.
    :type group_input: LayerOutput
    :params hidden_size: The size of the fully connected layer.
    :type hidden_size: int
    """
    conv3 = paddle.networks.sequence_conv_pool(
        input=group_input, context_len=3, hidden_size=hidden_size)
    conv4 = paddle.networks.sequence_conv_pool(
        input=group_input, context_len=4, hidden_size=hidden_size)

    fc_param_attr = paddle.attr.ParamAttr(name='_cov_value_weight')
    fc_bias_attr = paddle.attr.ParamAttr(name='_cov_value_bias')
    linear_proj = paddle.layer.fc(input=[conv3, conv4],
                                  size=hidden_size,
                                  param_attr=[fc_param_attr, fc_param_attr],
                                  bias_attr=fc_bias_attr,
                                  act=paddle.activation.Linear())

    return linear_proj


def nested_net(dict_dim, class_num, is_infer=False):
    """
    Nested network definition.
    :param dict_dim: Size of word dictionary.
    :type dict_dim: int
    :params class_num: Number of instance class.
    :type class_num: int
    :params is_infer: The boolean parameter
                        indicating inferring or training.
    :type is_infer: bool
    """
    data = paddle.layer.data(
        "word", paddle.data_type.integer_value_sub_sequence(dict_dim))

    emb = paddle.layer.embedding(input=data, size=conf.emb_size)
    nest_group = paddle.layer.recurrent_group(
        input=[paddle.layer.SubsequenceInput(emb), conf.hidden_size],
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
