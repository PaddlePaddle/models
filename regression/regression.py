"""
    This python script is a example model configuration for regression problem,
    based on PaddlePaddle V2 APIs
"""

import paddle.v2 as paddle
import data_process
import sys
import gzip
import numpy as np

word_emb_dim = 512
hidden_dim = 512
dict_size = 30000


def encode_net(input_dict_dim, is_static=False):
    """
        Define the encode networks, the input data is encoded to obtain the same size encoded vector

        :param input_dict_dim: size of dict
        :type input_dict_dim: int
        :param is_static: True or False
            True: The parameter will be set to fixed and will not be updated,
                      Used to encode the target data to obtain the same size of the vector
            False: The parameters will be updated during the training process
        :type is_static: int,

    """

    if is_static:
        input_name = "input2"
        _emb_name = "_emb_basic"
        _avg_name = '_avg.bias_basic'
        hidden_input_w = "_hidden_input.w2"
        hidden_input_bias = "_hidden_input.bias2"
    else:
        input_name = "input1"
        _emb_name = "_emb_inpu1"
        _avg_name = '_avg.bias_input1'
        hidden_input_w = "_hidden_input.w1"
        hidden_input_bias = "_hidden_input.bias1"

    input_data = paddle.layer.data(
        name=input_name,
        type=paddle.data_type.integer_value_sequence(input_dict_dim))

    input_emb = paddle.layer.embedding(
        input=input_data,
        size=word_emb_dim,
        param_attr=paddle.attr.Param(
            name=_emb_name, initial_std=0.02, is_static=is_static))

    input_vec = paddle.layer.pooling(
        input=input_emb,
        pooling_type=paddle.pooling.Sum(),
        bias_attr=paddle.attr.ParameterAttribute(
            name=_avg_name, initial_std=0.01, is_static=is_static))

    hidden_input = paddle.layer.fc(
        input=input_vec,
        size=hidden_dim,
        act=paddle.activation.Tanh(),
        param_attr=paddle.attr.Param(
            name=hidden_input_w, initial_std=0.03, is_static=is_static),
        bias_attr=paddle.attr.ParameterAttribute(
            name=hidden_input_bias, is_static=is_static))

    return hidden_input


def regression_net(input1_dict_dim, input2_dict_dim):

    # Network Architecture
    encode_input2 = encode_net(input1_dict_dim, is_static=True)
    encode_input1 = encode_net(input2_dict_dim, is_static=False)

    cost = paddle.layer.mse_cost(input=encode_input1, label=encode_input2)
    return cost


def load_parameter(file_name, h, w):
    with open(file_name, 'rb') as f:
        f.read(16)  # skip header.
        return np.fromfile(f, dtype=np.float32).reshape(h, w)


def main():
    paddle.init(use_gpu=False, trainer_count=1)

    #input1 and input2 dict dim
    input1_dict_dim = input2_dict_dim = dict_size

    #train the network
    cost = regression_net(input1_dict_dim, input2_dict_dim)
    parameters = paddle.parameters.create(cost)

    # initial the parameters of input2 networks
    parameters.set('_emb_basic',
                   load_parameter('./models/_emb_basic', dict_size,
                                  word_emb_dim))
    parameters.set('_avg.bias_basic',
                   load_parameter('./models/_avg.bias_basic', 1, word_emb_dim))
    parameters.set('_hidden_input.w2',
                   load_parameter('./models/_hidden_input.w2', word_emb_dim,
                                  hidden_dim))
    parameters.set('_hidden_input.bias2',
                   load_parameter('./models/_hidden_input.bias2', 1,
                                  hidden_dim))

    # define optimize method and trainer
    optimizer = paddle.optimizer.Adam(
        learning_rate=5e-5,
        regularization=paddle.optimizer.L2Regularization(rate=8e-4))

    trainer = paddle.trainer.SGD(
        cost=cost, parameters=parameters, update_equation=optimizer)

    # define data reader
    data_reader = paddle.batch(
        paddle.reader.shuffle(data_process.train(dict_size), buf_size=8192),
        batch_size=5)

    # define event_handler callback
    def event_handler(event):
        if isinstance(event, paddle.event.EndIteration):
            if event.batch_id % 100 == 0:
                print "\nPass %d, Batch %d, Cost %f" % (
                    event.pass_id, event.batch_id, event.cost)
        if isinstance(event, paddle.event.EndPass):
            with gzip.open('params_pass_%d.tar.gz' % event.pass_id, 'w') as f:
                parameters.to_tar(f)
            result = trainer.test(reader=paddle.batch(
                data_process.test(dict_size), batch_size=2))
            print "Test %d, Cost %f" % (event.pass_id, result.cost)

    # start to train
    trainer.train(reader=data_reader, event_handler=event_handler, num_passes=2)


if __name__ == '__main__':
    main()
