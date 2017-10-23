import os
import sys
import gzip
import functools
import paddle.v2 as paddle
import numpy as np
from metrics import ndcg
import argparse

# ranknet is the classic pairwise learning to rank algorithm
# http://icml.cc/2015/wp-content/uploads/2015/06/icml_ranking.pdf


def half_ranknet(name_prefix, input_dim):
    """
    parameter in same name will be shared in paddle framework,
    these parameters in ranknet can be used in shared state,
    e.g. left network and right network shared parameters in detail
    https://github.com/PaddlePaddle/Paddle/blob/develop/doc/design/api.md
    """
    # data layer
    data = paddle.layer.data(name_prefix + "/data",
                             paddle.data_type.dense_vector(input_dim))

    # hidden layer
    hd1 = paddle.layer.fc(
        input=data,
        size=10,
        act=paddle.activation.Tanh(),
        param_attr=paddle.attr.Param(initial_std=0.01, name="hidden_w1"))
    # fully connect layer/ output layer
    output = paddle.layer.fc(
        input=hd1,
        size=1,
        act=paddle.activation.Linear(),
        param_attr=paddle.attr.Param(initial_std=0.01, name="output"))
    return output


def ranknet(input_dim):
    # label layer
    label = paddle.layer.data("label", paddle.data_type.dense_vector(1))

    # reuse the parameter in half_ranknet
    output_left = half_ranknet("left", input_dim)
    output_right = half_ranknet("right", input_dim)

    evaluator = paddle.evaluator.auc(input=output_left, label=label)
    # rankcost layer
    cost = paddle.layer.rank_cost(
        name="cost", left=output_left, right=output_right, label=label)
    return cost


def train_ranknet(num_passes):
    train_reader = paddle.batch(
        paddle.reader.shuffle(paddle.dataset.mq2007.train, buf_size=100),
        batch_size=100)
    test_reader = paddle.batch(paddle.dataset.mq2007.test, batch_size=100)

    # mq2007 feature_dim = 46, dense format
    # fc hidden_dim = 128
    feature_dim = 46
    cost = ranknet(feature_dim)
    parameters = paddle.parameters.create(cost)

    trainer = paddle.trainer.SGD(
        cost=cost,
        parameters=parameters,
        update_equation=paddle.optimizer.Adam(learning_rate=2e-4))

    # Define the input data order
    feeding = {"label": 0, "left/data": 1, "right/data": 2}

    #  Define end batch and end pass event handler
    def event_handler(event):
        if isinstance(event, paddle.event.EndIteration):
            if event.batch_id % 100 == 0:
                print "Pass %d Batch %d Cost %.9f" % (
                    event.pass_id, event.batch_id, event.cost)
            else:
                sys.stdout.write(".")
                sys.stdout.flush()
        if isinstance(event, paddle.event.EndPass):
            result = trainer.test(reader=test_reader, feeding=feeding)
            print "\nTest with Pass %d, %s" % (event.pass_id, result.metrics)
            with gzip.open("ranknet_params_%d.tar.gz" % (event.pass_id),
                           "w") as f:
                trainer.save_parameter_to_tar(f)

    trainer.train(
        reader=train_reader,
        event_handler=event_handler,
        feeding=feeding,
        num_passes=num_passes)


def ranknet_infer(pass_id):
    """
  load the trained model. And predict with plain txt input
  """
    print "Begin to Infer..."
    feature_dim = 46

    # we just need half_ranknet to predict a rank score,
    # which can be used in sort documents
    output = half_ranknet("infer", feature_dim)
    parameters = paddle.parameters.Parameters.from_tar(
        gzip.open("ranknet_params_%d.tar.gz" % (pass_id)))

    # load data of same query and relevance documents,
    # need ranknet to rank these candidates
    infer_query_id = []
    infer_data = []
    infer_doc_index = []

    # convert to mq2007 built-in data format
    # <query_id> <relevance_score> <feature_vector>
    plain_txt_test = functools.partial(
        paddle.dataset.mq2007.test, format="plain_txt")

    for query_id, relevance_score, feature_vector in plain_txt_test():
        infer_query_id.append(query_id)
        infer_data.append([feature_vector])

    # predict score of infer_data document.
    # Re-sort the document base on predict score
    # in descending order. then we build the ranking documents
    scores = paddle.infer(
        output_layer=output, parameters=parameters, input=infer_data)
    print scores
    for query_id, score in zip(infer_query_id, scores):
        print "query_id : ", query_id, " ranknet rank document order : ", score


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ranknet demo')
    parser.add_argument("--run_type", type=str, help="run type is train|infer")
    parser.add_argument(
        "--num_passes",
        type=int,
        help="num of passes in train| infer pass number of model")
    args = parser.parse_args()
    paddle.init(use_gpu=False, trainer_count=4)
    if args.run_type == "train":
        train_ranknet(args.num_passes)
    elif args.run_type == "infer":
        ranknet_infer(pass_id=args.pass_num - 1)
