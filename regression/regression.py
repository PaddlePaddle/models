"""
    This python script is a example model configuration for regression problem,
    based on PaddlePaddle V2 APIs
"""

import paddle.v2 as paddle
import sys


def encode_net(input_dict_dim, word_emb_dim, hidden_dim, is_static=False):
    """
        the input data is encoded to obtain the same size encoded vector
        params:
            input_dict_dim: int, size of dict
            word_emb_dim: int, size of embedding layer
            hidden_dim: int, size of hidden layer
            is_static: bool, eg:True or False
                True: The parameter will be set to fixed and will not be updated,
                      Used to encode the target data to obtain the same size of the vector
                False: The parameters will be updated during the training process
    """

    if is_static:
        hidden_input_w = "_hidden_input.w2"
        hidden_input_bias = "_hidden_input.bias2"
    else:
        hidden_input_w = "_hidden_input.w1"
        hidden_input_bias = "_hidden_input.bias1"

    input_data = paddle.layer.data(
        name='input_word',
        type=paddle.data_type.integer_value_sequence(input_dict_dim))

    input_emb = paddle.layer.embedding(
        input=input_data,
        size=word_emb_dim,
        param_attr=paddle.attr.Param(name='_emb_basic', is_static=is_static))

    input_vec = paddle.layer.pooling(
        input=input_emb,
        pooling_type=paddle.pooling.Sum(),
        bias_attr=paddle.attr.ParameterAttribute(
            name='_avg.bias_basic', is_static=is_static))

    hidden_input = paddle.layer.fc(
        input=input_vec,
        size=hidden_dim,
        act=paddle.activation.Tanh(),
        param_attr=paddle.attr.Param(name=hidden_input_w, is_static=is_static),
        bias_attr=paddle.attr.ParameterAttribute(
            name=hidden_input_bias, is_static=is_static))

    return hidden_input


def regression_net(input1_dict_dim, input2_dict_dim):
    word_emb_dim = 512
    hidden_dim = 512

    # Network Architecture
    encode_input2 = encode_net(
        input1_dict_dim, word_emb_dim, hidden_dim, is_static=True)
    encode_input1 = encode_net(
        input2_dict_dim, word_emb_dim, hidden_dim, is_static=False)

    cost = paddle.layer.mse_cost(input=encode_input1, label=encode_input2)
    return cost


def main():
    paddle.init(use_gpu=False, trainer_count=1)

    #input1 and input2 dict dim
    dict_size = 30000
    input1_dict_dim = input2_dict_dim = dict_size

    #train the network
    cost = regression_net(input1_dict_dim, input2_dict_dim)
    parameters = paddle.parameters.create(cost)

    # define optimize method and trainer
    optimizer = paddle.optimizer.Adam(
        learning_rate=5e-5,
        regularization=paddle.optimizer.L2Regularization(rate=8e-4))

    trainer = paddle.trainer.SGD(
        cost=cost, parameters=parameters, update_equation=optimizer)

    # define data reader
    wmt14_reader = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.wmt14.train(dict_size), buf_size=8192),
        batch_size=5)

    # define event_handler callback
    def event_handler(event):
        if isinstance(event, paddle.event.EndIteration):
            if event.batch_id % 100 == 0:
                print "\nPass %d, Batch %d, Cost %f" % (
                    event.pass_id, event.batch_id, event.cost)
        if isinstance(event, paddle.event.EndPass):
            result = trainer.test(reader=paddle.batch(
                paddle.dataset.wmt14.test(dict_size), batch_size=2))
            print "Test %d, Cost %f" % (event.pass_id, result.cost)

    # start to train
    trainer.train(
        reader=wmt14_reader, event_handler=event_handler, num_passes=2)


if __name__ == '__main__':
    main()
