import os
import sys
import gzip
import click

import paddle.v2 as paddle

import reader
from network_conf import nest_net
from utils import build_dict, load_dict, logger


@click.command('train')
@click.option(
    "--train_data_dir",
    default=None,
    help=("path of training dataset (default: None). "
          "if this parameter is not set, "
          "imdb dataset will be used."))
@click.option(
    "--test_data_dir",
    default=None,
    help=("path of testing dataset (default: None). "
          "if this parameter is not set, "
          "imdb dataset will be used."))
@click.option(
    "--word_dict_path",
    type=str,
    default=None,
    help=("path of word dictionary (default: None)."
          "if this parameter is not set, imdb dataset will be used."
          "if this parameter is set, but the file does not exist, "
          "word dictionay will be built from "
          "the training data automatically."))
@click.option(
    "--class_num", type=int, default=2, help="class number (default: 2).")
@click.option(
    "--batch_size",
    type=int,
    default=32,
    help=("the number of training examples in one batch "
          "(default: 32)."))
@click.option(
    "--num_passes",
    type=int,
    default=10,
    help="number of passes to train (default: 10).")
@click.option(
    "--model_save_dir",
    type=str,
    default="models",
    help="path to save the trained models (default: 'models').")
def train(train_data_dir, test_data_dir, word_dict_path, class_num,
          model_save_dir, batch_size, num_passes):
    """
    :params train_data_path: path of training data, if this parameter
        is not specified, imdb dataset will be used to run this example
    :type train_data_path: str
    :params test_data_path: path of testing data, if this parameter
        is not specified, imdb dataset will be used to run this example
    :type test_data_path: str
    :params word_dict_path: path of training data, if this parameter
        is not specified, imdb dataset will be used to run this example
    :type word_dict_path: str
    :params model_save_dir: dir where models saved
    :type num_pass: str
    :params batch_size: train batch size
    :type num_pass: int
    :params num_pass: train pass number
    :type num_pass: int
    """
    if train_data_dir is not None:
        assert word_dict_path, ("the parameter train_data_dir, word_dict_path "
                                "should be set at the same time.")

    if not os.path.exists(model_save_dir):
        os.mkdir(model_save_dir)

    use_default_data = (train_data_dir is None)

    if use_default_data:
        logger.info(("No training data are porivided, "
                     "use imdb to train the model."))
        logger.info("please wait to build the word dictionary ...")

        word_dict = reader.imdb_word_dict()

        train_reader = paddle.batch(
            paddle.reader.shuffle(
                lambda: reader.imdb_train(word_dict), buf_size=1000),
            batch_size=100)
        test_reader = paddle.batch(
            lambda: reader.imdb_test(word_dict), batch_size=100)
        class_num = 2
    else:
        if word_dict_path is None or not os.path.exists(word_dict_path):
            logger.info(("word dictionary is not given, the dictionary "
                         "is automatically built from the training data."))

            # build the word dictionary to map the original string-typed
            # words into integer-typed index
            build_dict(
                data_dir=train_data_dir,
                save_path=word_dict_path,
                use_col=1,
                cutoff_fre=0)

        word_dict = load_dict(word_dict_path)
        class_num = class_num
        logger.info("class number is : %d." % class_num)

        train_reader = paddle.batch(
            paddle.reader.shuffle(
                reader.train_reader(train_data_dir, word_dict), buf_size=1000),
            batch_size=batch_size)

        if test_data_dir is not None:
            # here, because training and testing data share a same format,
            # we still use the reader.train_reader to read the testing data.
            test_reader = paddle.batch(
                paddle.reader.shuffle(
                    reader.train_reader(test_data_dir, word_dict),
                    buf_size=1000),
                batch_size=batch_size)
        else:
            test_reader = None

    dict_dim = len(word_dict)
    emb_size = 28
    hidden_size = 128

    logger.info("length of word dictionary is : %d." % (dict_dim))

    paddle.init(use_gpu=True, trainer_count=4)

    # network config
    cost, prob, label = nest_net(
        dict_dim, emb_size, hidden_size, class_num, is_infer=False)

    # create parameters
    parameters = paddle.parameters.create(cost)

    # create optimizer
    adam_optimizer = paddle.optimizer.Adam(
        learning_rate=1e-3,
        regularization=paddle.optimizer.L2Regularization(rate=1e-3),
        model_average=paddle.optimizer.ModelAverage(average_window=0.5))

    # create trainer
    trainer = paddle.trainer.SGD(
        cost=cost,
        extra_layers=paddle.evaluator.auc(input=prob, label=label),
        parameters=parameters,
        update_equation=adam_optimizer)

    # begin training network
    feeding = {"word": 0, "label": 1}

    def _event_handler(event):
        """
        Define end batch and end pass event handler
        """
        if isinstance(event, paddle.event.EndIteration):
            if event.batch_id % 100 == 0:
                logger.info("Pass %d, Batch %d, Cost %f, %s\n" % (
                    event.pass_id, event.batch_id, event.cost, event.metrics))

        if isinstance(event, paddle.event.EndPass):
            if test_reader is not None:
                result = trainer.test(reader=test_reader, feeding=feeding)
                logger.info("Test at Pass %d, %s \n" % (event.pass_id,
                                                        result.metrics))
            with gzip.open(
                    os.path.join(model_save_dir, "params_pass_%05d.tar.gz" %
                                 event.pass_id), "w") as f:
                parameters.to_tar(f)

    trainer.train(
        reader=train_reader,
        event_handler=_event_handler,
        feeding=feeding,
        num_passes=num_passes)

    logger.info("Training has finished.")


if __name__ == "__main__":
    train()
