import os
import gzip
import functools
import argparse
import logging
import numpy as np

import paddle.v2 as paddle

from ranknet import ranknet
from lambda_rank import lambda_rank

logger = logging.getLogger("paddle")
logger.setLevel(logging.INFO)


def ranknet_train(input_dim, num_passes, model_save_dir):
    train_reader = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.mq2007.train, buf_size=100),
        batch_size=100)
    test_reader = paddle.batch(paddle.dataset.mq2007.test, batch_size=100)

    cost = ranknet(input_dim)
    parameters = paddle.parameters.create(cost)

    trainer = paddle.trainer.SGD(
        cost=cost,
        parameters=parameters,
        update_equation=paddle.optimizer.Adam(learning_rate=2e-4))

    feeding = {"label": 0, "left_data": 1, "right_data": 2}

    def score_diff(right_score, left_score):
        return np.average(np.abs(right_score - left_score))

    #  Define end batch and end pass event handler
    def event_handler(event):
        if isinstance(event, paddle.event.EndIteration):
            if event.batch_id % 25 == 0:
                diff = score_diff(
                    event.gm.getLayerOutputs("left_score")["left_score"][
                        "value"],
                    event.gm.getLayerOutputs("right_score")["right_score"][
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


def lambda_rank_train(input_dim, num_passes, model_save_dir):
    # The input for LambdaRank must be a sequence.
    fill_default_train = functools.partial(
        paddle.dataset.mq2007.train, format="listwise")
    fill_default_test = functools.partial(
        paddle.dataset.mq2007.test, format="listwise")

    train_reader = paddle.batch(
        paddle.reader.shuffle(
            fill_default_train, buf_size=100), batch_size=32)
    test_reader = paddle.batch(fill_default_test, batch_size=32)

    cost = lambda_rank(input_dim)
    parameters = paddle.parameters.create(cost)

    trainer = paddle.trainer.SGD(
        cost=cost,
        parameters=parameters,
        update_equation=paddle.optimizer.Adam(learning_rate=1e-4))

    feeding = {"label": 0, "data": 1}

    #  Define end batch and end pass event handler.
    def event_handler(event):
        if isinstance(event, paddle.event.EndIteration):
            logger.info("Pass %d Batch %d Cost %.9f" %
                        (event.pass_id, event.batch_id, event.cost))
        if isinstance(event, paddle.event.EndPass):
            result = trainer.test(reader=test_reader, feeding=feeding)
            logger.info("\nTest with Pass %d, %s" %
                        (event.pass_id, result.metrics))
            with gzip.open(
                    os.path.join(model_save_dir, "lambda_rank_params_%d.tar.gz"
                                 % (event.pass_id)), "w") as f:
                trainer.save_parameter_to_tar(f)

    trainer.train(
        reader=train_reader,
        event_handler=event_handler,
        feeding=feeding,
        num_passes=num_passes)


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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not os.path.exists(args.model_save_dir): os.mkdir(args.model_save_dir)

    paddle.init(use_gpu=args.use_gpu, trainer_count=args.trainer_count)

    # Training dataset: mq2007, input_dim = 46, dense format.
    input_dim = 46

    if args.model_type == "ranknet":
        ranknet_train(input_dim, args.num_passes, args.model_save_dir)
    elif args.model_type == "lambdarank":
        lambda_rank_train(input_dim, args.num_passes, args.model_save_dir)
    else:
        logger.fatal(("A wrong value for parameter model type. "
                      "Available options are: ranknet or lambdarank."))
