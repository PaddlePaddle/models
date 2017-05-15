import os, sys
import gzip
import sqlite3
import paddle.v2 as paddle
import numpy as np
import functools

#lambdaRank is listwise learning to rank model


def lambdaRank(input_dim):
    label = paddle.layer.data("label",
                              paddle.data_type.dense_vector_sequence(1))
    data = paddle.layer.data("data",
                             paddle.data_type.dense_vector_sequence(input_dim))

    # hidden layer
    hd1 = paddle.layer.fc(
        input=data,
        size=10,
        act=paddle.activation.Tanh(),
        param_attr=paddle.attr.Param(initial_std=0.01))
    output = paddle.layer.fc(
        input=hd1,
        size=1,
        act=paddle.activation.Linear(),
        param_attr=paddle.attr.Param(initial_std=0.01))
    cost = paddle.layer.lambda_cost(
        input=output, score=label, NDCG_num=6, max_sort_size=-1)
    return cost, output


def train_lambdaRank(num_passes):
    # listwise input sequence
    fill_default_train = functools.partial(
        paddle.dataset.mq2007.train, format="listwise")
    fill_default_test = functools.partial(
        paddle.dataset.mq2007.test, format="listwise")
    train_reader = paddle.batch(
        paddle.reader.shuffle(fill_default_train, buf_size=100), batch_size=32)
    test_reader = paddle.batch(
        paddle.reader.shuffle(fill_default_test, buf_size=100), batch_size=32)

    # mq2007 input_dim = 46, dense format 
    input_dim = 46
    cost, output = lambdaRank(input_dim)
    parameters = paddle.parameters.create(cost)

    trainer = paddle.trainer.SGD(
        cost=cost,
        parameters=parameters,
        update_equation=paddle.optimizer.Adam(learning_rate=1e-4))

    #  Define end batch and end pass event handler
    def event_handler(event):
        if isinstance(event, paddle.event.EndIteration):
            print "Pass %d Batch %d Cost %.9f" % (event.pass_id, event.batch_id,
                                                  event.cost)
        if isinstance(event, paddle.event.EndPass):
            result = trainer.test(reader=test_reader, feeding=feeding)
            print "\nTest with Pass %d, %s" % (event.pass_id, result.metrics)
            with gzip.open("lambdaRank_params_%d.tar.gz" % (event.pass_id),
                           "w") as f:
                parameters.to_tar(f)

    feeding = {"label": 0, "data": 1}
    trainer.train(
        reader=train_reader,
        event_handler=event_handler,
        feeding=feeding,
        num_passes=num_passes)


def lambdaRank_infer(pass_id):
    """
  lambdaRank model inference interface
  parameters:
    pass_id : inference model in pass_id
  """
    print "Begin to Infer..."
    input_dim = 46
    output = lambdaRank(input_dim)
    parameters = paddle.parameters.Parameters.from_tar(
        gzip.open("lambdaRank_params_%d.tar.gz" % (pass_id - 1)))

    infer_query_id = None
    infer_data = []
    infer_data_num = 1000
    fill_default_test = functools.partial(
        paddle.dataset.mq2007.test, format="listwise")
    for label, querylist in fill_default_test():
        infer_data.append(querylist)
        if len(infer_data) == infer_data_num:
            break
    predicitons = paddle.infer(
        output_layer=output, parameters=parameters, input=infer_data)
    for i, score in enumerate(predicitons):
        print score


if __name__ == '__main__':
    paddle.init(use_gpu=False, trainer_count=4)
    train_lambdaRank(100)
    lambdaRank_infer(pass_id=2)
