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


def lambda_rank(input_dim, is_infer):
    """
    LambdaRank is a listwise rank model, the input data and label
    must be sequences.

    https://papers.nips.cc/paper/2971-learning-to-rank-with-nonsmooth-cost-functions.pdf
    parameters :
      input_dim, one document's dense feature vector dimension

    The format of the dense_vector_sequence is as follows:
    [[f, ...], [f, ...], ...], f is a float or an int number
    """
    if not is_infer:
        label = paddle.layer.data("label",
                                  paddle.data_type.dense_vector_sequence(1))
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
        # Define the cost layer.
        cost = paddle.layer.lambda_cost(
            input=output, score=label, NDCG_num=6, max_sort_size=-1)
        return cost, output
    return output


def lambda_rank_train(num_passes, model_save_dir):
    # The input for LambdaRank must be a sequence.
    fill_default_train = functools.partial(
        paddle.dataset.mq2007.train, format="listwise")
    fill_default_test = functools.partial(
        paddle.dataset.mq2007.test, format="listwise")

    train_reader = paddle.batch(
        paddle.reader.shuffle(
            fill_default_train, buf_size=100), batch_size=32)
    test_reader = paddle.batch(fill_default_test, batch_size=32)

    # Training dataset: mq2007, input_dim = 46, dense format.
    input_dim = 46
    cost, output = lambda_rank(input_dim, is_infer=False)
    parameters = paddle.parameters.create(cost)

    trainer = paddle.trainer.SGD(
        cost=cost,
        parameters=parameters,
        update_equation=paddle.optimizer.Adam(learning_rate=1e-4))

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

    feeding = {"label": 0, "data": 1}
    trainer.train(
        reader=train_reader,
        event_handler=event_handler,
        feeding=feeding,
        num_passes=num_passes)


def lambda_rank_infer(test_model_path):
    """LambdaRank model inference interface.

    Parameters:
        test_model_path : The path of the trained model.
    """
    logger.info("Begin to Infer...")
    input_dim = 46
    output = lambda_rank(input_dim, is_infer=True)
    parameters = paddle.parameters.Parameters.from_tar(
        gzip.open(test_model_path))

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="PaddlePaddle LambdaRank example.")
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
    paddle.init(use_gpu=args.use_gpu, trainer_count=args.trainer_count)
    if args.run_type == "train":
        lambda_rank_train(args.num_passes, args.model_save_dir)
    elif args.run_type == "infer":
        assert os.path.exists(args.test_model_path), (
            "The trained model does not exit. Please set a correct path.")
        lambda_rank_infer(args.test_model_path)
    else:
        logger.fatal(("A wrong value for parameter run type. "
                      "Available options are: train or infer."))
