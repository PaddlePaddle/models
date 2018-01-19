"""
ranknet is the classic pairwise learning to rank algorithm
http://icml.cc/2015/wp-content/uploads/2015/06/icml_ranking.pdf
"""
import paddle.v2 as paddle


def half_ranknet(name_prefix, input_dim):
    """
    parameter in same name will be shared in paddle framework,
    these parameters in ranknet can be used in shared state,
    e.g. left network and right network shared parameters in detail
    https://github.com/PaddlePaddle/Paddle/blob/develop/doc/design/api.md
    """
    # data layer
    data = paddle.layer.data(name_prefix + "_data",
                             paddle.data_type.dense_vector(input_dim))

    # hidden layer
    hd1 = paddle.layer.fc(input=data,
                          name=name_prefix + "_hidden",
                          size=10,
                          act=paddle.activation.Tanh(),
                          param_attr=paddle.attr.Param(
                              initial_std=0.01, name="hidden_w1"))

    # fully connected layer and output layer
    output = paddle.layer.fc(input=hd1,
                             name=name_prefix + "_score",
                             size=1,
                             act=paddle.activation.Linear(),
                             param_attr=paddle.attr.Param(
                                 initial_std=0.01, name="output"))
    return output


def ranknet(input_dim):
    # label layer
    label = paddle.layer.data("label", paddle.data_type.dense_vector(1))

    # reuse the parameter in half_ranknet
    output_left = half_ranknet("left", input_dim)
    output_right = half_ranknet("right", input_dim)

    # rankcost layer
    cost = paddle.layer.rank_cost(
        name="cost", left=output_left, right=output_right, label=label)
    return cost
