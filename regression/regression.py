import paddle.v2 as paddle
import sys


def regression_net(input1_dict_dim, input2_dict_dim, is_generating=False):
    ### Network Architecture
    word_vector_dim = 512
    word_emb_dim = 512
    hidden_dim = 512

    #input2 architecture
    input2 = paddle.layer.data(
        name='input2_word',
        type=paddle.data_type.integer_value_sequence(input2_dict_dim))

    with paddle.layer.mixed(size=word_emb_dim, bias_attr=False) as input2_emb:
        input2_emb += paddle.layer.table_projection(
            input=input2,
            param_attr=paddle.attr.Param(name='_emb_basic', is_static=True))

    input2_vec = paddle.layer.pooling(
        input=input2_emb,
        pooling_type=paddle.pooling.Sum(),
        #act=paddle.activation.Tanh(),
        bias_attr=paddle.attr.ParameterAttribute(
            name='_avg.bias_basic', is_static=True))

    hidden_input2 = paddle.layer.fc(
        input=input2_vec,
        size=hidden_dim,
        act=paddle.activation.Tanh(),
        param_attr=paddle.attr.Param(name='_hidden_input2.w', is_static=True),
        bias_attr=paddle.attr.ParameterAttribute(
            name='_hidden_input2.bias', is_static=True))

    #input1 architecture
    input1 = paddle.layer.data(
        name='input1_word',
        type=paddle.data_type.integer_value_sequence(input1_dict_dim))

    with paddle.layer.mixed(size=word_emb_dim, bias_attr=False) as input1_emb:
        input1_emb += paddle.layer.table_projection(
            input=input1,
            param_attr=paddle.attr.Param(name='emb_input1', initial_std=0.02))

    input1_vec = paddle.layer.pooling(
        input=input1_emb,
        pooling_type=paddle.pooling.Sum(),
        #act=paddle.activation.Tanh(),
        bias_attr=paddle.attr.ParameterAttribute(
            name='_avg.bias_input1', initial_std=0.01))

    hidden_input1 = paddle.layer.fc(
        input=input1_vec,
        size=hidden_dim,
        act=paddle.activation.Tanh(),
        param_attr=paddle.attr.ParameterAttribute(
            name='_hidden_input1.w', initial_std=0.03),
        bias_attr=paddle.attr.ParameterAttribute(name='_hidden_input1.bias'))

    cost = paddle.layer.mse_cost(input=hidden_input1, label=hidden_input2)
    #cost = paddle.layer.huber_cost(input=hidden_input1, label=hidden_input2)
    return cost


def main():
    paddle.init(use_gpu=False, trainer_count=1)
    is_generating = False

    #input1 and input2 dict dim
    dict_size = 30000
    input1_dict_dim = input2_dict_dim = dict_size

    #train the network
    if not is_generating:

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
                    print "\nPass %d, Batch %d, Cost %f, %s" % (
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
