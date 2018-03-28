"""
LambdaRank is a listwise rank model.
https://papers.nips.cc/paper/2971-learning-to-rank-with-nonsmooth-cost-functions.pdf
"""
import paddle.v2 as paddle


def lambda_rank(input_dim, is_infer=False):
    """
    The input data and label for LambdaRank must be sequences.

    parameters :
      input_dim, one document's dense feature vector dimension

    The format of the dense_vector_sequence is as follows:
    [[f, ...], [f, ...], ...], f is a float or an int number
    """
    data = paddle.layer.data("data",
                             paddle.data_type.dense_vector_sequence(input_dim))

    # Define the hidden layer.
    hd1 = paddle.layer.fc(input=data,
                          size=128,
                          act=paddle.activation.Tanh(),
                          param_attr=paddle.attr.Param(initial_std=0.01))

    hd2 = paddle.layer.fc(input=hd1,
                          size=10,
                          act=paddle.activation.Tanh(),
                          param_attr=paddle.attr.Param(initial_std=0.01))
    output = paddle.layer.fc(input=hd2,
                             size=1,
                             act=paddle.activation.Linear(),
                             param_attr=paddle.attr.Param(initial_std=0.01))

    if not is_infer:
        label = paddle.layer.data("label",
                                  paddle.data_type.dense_vector_sequence(1))

        cost = paddle.layer.lambda_cost(
            input=output, score=label, NDCG_num=6, max_sort_size=-1)
        return cost
    else:
        return output
