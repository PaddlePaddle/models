import os
import sys
import gzip
import functools
import argparse
import logging
import numpy as np

import paddle.v2 as paddle

logger = logging.getLogger("paddle")
logger.setLevel(logging.INFO)

# ranknet is the classic pairwise learning to rank algorithm
# http://icml.cc/2015/wp-content/uploads/2015/06/icml_ranking.pdf


def score_diff(right_score, left_score):
    return np.average(np.abs(right_score - left_score))


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
    hd1 = paddle.layer.fc(
        input=data,
        name=name_prefix + "_hidden",
        size=10,
        act=paddle.activation.Tanh(),
        param_attr=paddle.attr.Param(initial_std=0.01, name="hidden_w1"))

    # fully connected layer and output layer
    output = paddle.layer.fc(
        input=hd1,
        name=name_prefix + "_score",
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

    # rankcost layer
    cost = paddle.layer.rank_cost(
        name="cost", left=output_left, right=output_right, label=label)
    return cost


def ranknet_train(num_passes, model_save_dir):
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
    feeding = {"label": 0, "left_data": 1, "right_data": 2}

    #  Define end batch and end pass event handler
    def event_handler(event):
        if isinstance(event, paddle.event.EndIteration):
            if event.batch_id % 25 == 0:
                diff = score_diff(
                    event.gm.getLayerOutputs("right_score")["right_score"][
                        "value"],
                    event.gm.getLayerOutputs("left_score")["left_score"][
                        "value"])
                logger.info(("Pass %d Batch %d : Cost %.6f, "
                             "average absolute diff scores: %.6f") %
                            (event.pass_id, event.batch_id, event.cost, diff))

        if isinstance(event, paddle.event.EndPass):
            result = trainer.test(reader=test_reader, feeding=feeding)
            logger.info("\nTest with Pass %d, %s" %
                        (event.pass_id, result.metrics))
            with gzip.open(
                    os.path.join(model_save_dir, "ranknet_params_%d.tar.gz" %
                                 (event.pass_id)), "w") as f:
                trainer.save_parameter_to_tar(f)

    trainer.train(
        reader=train_reader,
        event_handler=event_handler,
        feeding=feeding,
        num_passes=num_passes)


def ranknet_infer(model_path):
    """
    load the trained model. And predict with plain txt input
    """
    logger.info("Begin to Infer...")
    feature_dim = 46

    # we just need half_ranknet to predict a rank score,
    # which can be used in sort documents
    output = half_ranknet("infer", feature_dim)
    parameters = paddle.parameters.Parameters.from_tar(gzip.open(model_path))

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PaddlePaddle RankNet example.")
    parser.add_argument(
        "--run_type",
        type=str,
        help=("A flag indicating to run the training or the inferring task. "
              "Available options are: train or infer."),
        default="train")
    parser.add_argument(
        "--num_passes",
        type=int,
        help="The number of passes to train the model.",
        default=10)
    parser.add_argument(
        "--use_gpu",
        type=bool,
        help="A flag indicating whether to use the GPU device in training.",
        default=False)
    parser.add_argument(
        "--trainer_count",
        type=int,
        help="The thread number used in training.",
        default=1)
    parser.add_argument(
        "--model_save_dir",
        type=str,
        required=False,
        help=("The path to save the trained models."),
        default="models")
    parser.add_argument(
        "--test_model_path",
        type=str,
        required=False,
        help=("This parameter works only in inferring task to "
              "specify path of a trained model."),
        default="")

    args = parser.parse_args()
    if not os.path.exists(args.model_save_dir): os.mkdir(args.model_save_dir)

    paddle.init(use_gpu=args.use_gpu, trainer_count=args.trainer_count)

    if args.run_type == "train":
        ranknet_train(args.num_passes, args.model_save_dir)
    elif args.run_type == "infer":
        assert os.path.exists(
            args.test_model_path), "The trained model does not exit."
        ranknet_infer(args.test_model_path)
    else:
        logger.fatal(("A wrong value for parameter run type. "
                      "Available options are: train or infer."))
