import os
import gzip
import functools
import argparse

import paddle.v2 as paddle

from ranknet import half_ranknet
from lambda_rank import lambda_rank


def ranknet_infer(input_dim, model_path):
    """
    RankNet model inference interface.
    """
    # we just need half_ranknet to predict a rank score,
    # which can be used in sort documents
    output = half_ranknet("right", input_dim)
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
    for query_id, score in zip(infer_query_id, scores):
        print "query_id : ", query_id, " score : ", score


def lambda_rank_infer(input_dim, model_path):
    """
    LambdaRank model inference interface.
    """
    output = lambda_rank(input_dim, is_infer=True)
    parameters = paddle.parameters.Parameters.from_tar(gzip.open(model_path))

    infer_query_id = None
    infer_data = []
    infer_data_num = 1

    fill_default_test = functools.partial(
        paddle.dataset.mq2007.test, format="listwise")
    for label, querylist in fill_default_test():
        infer_data.append([querylist])
        if len(infer_data) == infer_data_num:
            break

    # Predict score of infer_data document.
    # Re-sort the document base on predict score.
    # In descending order. then we build the ranking documents.
    predicitons = paddle.infer(
        output_layer=output, parameters=parameters, input=infer_data)
    for i, score in enumerate(predicitons):
        print i, score


def parse_args():
    parser = argparse.ArgumentParser(
        description="PaddlePaddle learning to rank example.")
    parser.add_argument(
        "--model_type",
        type=str,
        help=("A flag indicating to run the RankNet or the LambdaRank model. "
              "Available options are: ranknet or lambdarank."),
        default="ranknet")
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
        "--test_model_path",
        type=str,
        required=True,
        help=("The path of a trained model."))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    assert os.path.exists(args.test_model_path), (
        "The trained model does not exit. Please set a correct path.")

    paddle.init(use_gpu=args.use_gpu, trainer_count=args.trainer_count)

    # Training dataset: mq2007, input_dim = 46, dense format.
    input_dim = 46

    if args.model_type == "ranknet":
        ranknet_infer(input_dim, args.test_model_path)
    elif args.model_type == "lambdarank":
        lambda_rank_infer(input_dim, args.test_model_path)
    else:
        logger.fatal(("A wrong value for parameter model type. "
                      "Available options are: ranknet or lambdarank."))
