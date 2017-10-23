import os
import sys
import gzip
import functools
import argparse
import numpy as np

import paddle.v2 as paddle


def lambda_rank(input_dim):
    """
    lambda_rank is a Listwise rank model, the input data and label
    must be sequences.

    https://papers.nips.cc/paper/2971-learning-to-rank-with-nonsmooth-cost-functions.pdf
    parameters :
      input_dim, one document's dense feature vector dimension

    format of the dense_vector_sequence:
    [[f, ...], [f, ...], ...], f is a float or an int number
    """

    label = paddle.layer.data("label",
                              paddle.data_type.dense_vector_sequence(1))
    data = paddle.layer.data("data",
                             paddle.data_type.dense_vector_sequence(input_dim))

    # hidden layer
    hd1 = paddle.layer.fc(
        input=data,
        size=128,
        act=paddle.activation.Tanh(),
        param_attr=paddle.attr.Param(initial_std=0.01))

    hd2 = paddle.layer.fc(
        input=hd1,
        size=10,
        act=paddle.activation.Tanh(),
        param_attr=paddle.attr.Param(initial_std=0.01))
    output = paddle.layer.fc(
        input=hd2,
        size=1,
        act=paddle.activation.Linear(),
        param_attr=paddle.attr.Param(initial_std=0.01))

    # evaluator
    evaluator = paddle.evaluator.auc(input=output, label=label)
    # cost layer
    cost = paddle.layer.lambda_cost(
        input=output, score=label, NDCG_num=6, max_sort_size=-1)
    return cost, output


def train_lambda_rank(num_passes):
    # listwise input sequence
    fill_default_train = functools.partial(
        paddle.dataset.mq2007.train, format="listwise")
    fill_default_test = functools.partial(
        paddle.dataset.mq2007.test, format="listwise")
    train_reader = paddle.batch(
        paddle.reader.shuffle(fill_default_train, buf_size=100), batch_size=32)
    test_reader = paddle.batch(fill_default_test, batch_size=32)

    # mq2007 input_dim = 46, dense format
    input_dim = 46
    cost, output = lambda_rank(input_dim)
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
            with gzip.open("lambda_rank_params_%d.tar.gz" % (event.pass_id),
                           "w") as f:
                trainer.save_parameter_to_tar(f)

    feeding = {"label": 0, "data": 1}
    trainer.train(
        reader=train_reader,
        event_handler=event_handler,
        feeding=feeding,
        num_passes=num_passes)


def lambda_rank_infer(pass_id):
    """lambda_rank model inference interface

    parameters:
        pass_id : inference model in pass_id
    """
    print "Begin to Infer..."
    input_dim = 46
    output = lambda_rank(input_dim)
    parameters = paddle.parameters.Parameters.from_tar(
        gzip.open("lambda_rank_params_%d.tar.gz" % (pass_id - 1)))

    infer_query_id = None
    infer_data = []
    infer_data_num = 1
    fill_default_test = functools.partial(
        paddle.dataset.mq2007.test, format="listwise")
    for label, querylist in fill_default_test():
        infer_data.append(querylist)
        if len(infer_data) == infer_data_num:
            break

    # predict score of infer_data document.
    # Re-sort the document base on predict score
    # in descending order. then we build the ranking documents
    predicitons = paddle.infer(
        output_layer=output, parameters=parameters, input=infer_data)
    for i, score in enumerate(predicitons):
        print i, score


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LambdaRank demo')
    parser.add_argument("--run_type", type=str, help="run type is train|infer")
    parser.add_argument(
        "--num_passes",
        type=int,
        help="num of passes in train| infer pass number of model")
    args = parser.parse_args()
    paddle.init(use_gpu=False, trainer_count=1)
    if args.run_type == "train":
        train_lambda_rank(args.num_passes)
    elif args.run_type == "infer":
        lambda_rank_infer(pass_id=args.num_passes - 1)
