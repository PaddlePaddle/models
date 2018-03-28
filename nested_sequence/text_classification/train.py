import os
import sys
import gzip
import click

import paddle.v2 as paddle

import reader
from network_conf import nested_net
from utils import build_word_dict, build_label_dict, load_dict, logger
from config import TrainerConfig as conf


@click.command('train')
@click.option(
    "--train_data_dir",
    default=None,
    help=("The path of training dataset (default: None). "
          "If this parameter is not set, "
          "imdb dataset will be used."))
@click.option(
    "--test_data_dir",
    default=None,
    help=("The path of testing dataset (default: None). "
          "If this parameter is not set, "
          "imdb dataset will be used."))
@click.option(
    "--word_dict_path",
    type=str,
    default=None,
    help=("The path of word dictionary (default: None). "
          "If this parameter is not set, imdb dataset will be used. "
          "If this parameter is set, but the file does not exist, "
          "word dictionay will be built from "
          "the training data automatically."))
@click.option(
    "--label_dict_path",
    type=str,
    default=None,
    help=("The path of label dictionary (default: None). "
          "If this parameter is not set, imdb dataset will be used. "
          "If this parameter is set, but the file does not exist, "
          "label dictionay will be built from "
          "the training data automatically."))
@click.option(
    "--model_save_dir",
    type=str,
    default="models",
    help="The path to save the trained models (default: 'models').")
def train(train_data_dir, test_data_dir, word_dict_path, label_dict_path,
          model_save_dir):
    """
    :params train_data_path: The path of training data, if this parameter
        is not specified, imdb dataset will be used to run this example
    :type train_data_path: str
    :params test_data_path: The path of testing data, if this parameter
        is not specified, imdb dataset will be used to run this example
    :type test_data_path: str
    :params word_dict_path: The path of word dictionary, if this parameter
        is not specified, imdb dataset will be used to run this example
    :type word_dict_path: str
    :params label_dict_path: The path of label dictionary, if this parameter
        is not specified, imdb dataset will be used to run this example
    :type label_dict_path: str
    :params model_save_dir: dir where models saved
    :type model_save_dir: str
    """
    if train_data_dir is not None:
        assert word_dict_path and label_dict_path, (
            "The parameter train_data_dir, word_dict_path, label_dict_path "
            "should be set at the same time.")

    if not os.path.exists(model_save_dir):
        os.mkdir(model_save_dir)

    use_default_data = (train_data_dir is None)

    if use_default_data:
        logger.info(("No training data are porivided, "
                     "use imdb to train the model."))
        logger.info("Please wait to build the word dictionary ...")

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
            logger.info(("Word dictionary is not given, the dictionary "
                         "is automatically built from the training data."))

            # build the word dictionary to map the original string-typed
            # words into integer-typed index
            build_word_dict(
                data_dir=train_data_dir,
                save_path=word_dict_path,
                use_col=1,
                cutoff_fre=0)

        if not os.path.exists(label_dict_path):
            logger.info(("Label dictionary is not given, the dictionary "
                         "is automatically built from the training data."))
            # build the label dictionary to map the original string-typed
            # label into integer-typed index
            build_label_dict(
                data_dir=train_data_dir, save_path=label_dict_path, use_col=0)

        word_dict = load_dict(word_dict_path)
        label_dict = load_dict(label_dict_path)

        class_num = len(label_dict)
        logger.info("Class number is : %d." % class_num)

        train_reader = paddle.batch(
            paddle.reader.shuffle(
                reader.train_reader(train_data_dir, word_dict, label_dict),
                buf_size=conf.buf_size),
            batch_size=conf.batch_size)

        if test_data_dir is not None:
            # here, because training and testing data share a same format,
            # we still use the reader.train_reader to read the testing data.
            test_reader = paddle.batch(
                paddle.reader.shuffle(
                    reader.train_reader(test_data_dir, word_dict, label_dict),
                    buf_size=conf.buf_size),
                batch_size=conf.batch_size)
        else:
            test_reader = None

    dict_dim = len(word_dict)

    logger.info("Length of word dictionary is : %d." % (dict_dim))

    paddle.init(use_gpu=conf.use_gpu, trainer_count=conf.trainer_count)

    # create optimizer
    adam_optimizer = paddle.optimizer.Adam(
        learning_rate=conf.learning_rate,
        regularization=paddle.optimizer.L2Regularization(
            rate=conf.l2_learning_rate),
        model_average=paddle.optimizer.ModelAverage(
            average_window=conf.average_window))

    # define network topology.
    cost, prob, label = nested_net(dict_dim, class_num, is_infer=False)

    # create all the trainable parameters.
    parameters = paddle.parameters.create(cost)

    # create the trainer instance.
    trainer = paddle.trainer.SGD(
        cost=cost,
        extra_layers=paddle.evaluator.auc(input=prob, label=label),
        parameters=parameters,
        update_equation=adam_optimizer)

    # feeding dictionary
    feeding = {"word": 0, "label": 1}

    def _event_handler(event):
        """
        Define the end batch and the end pass event handler.
        """
        if isinstance(event, paddle.event.EndIteration):
            if event.batch_id % conf.log_period == 0:
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
                trainer.save_parameter_to_tar(f)

    # begin training network
    trainer.train(
        reader=train_reader,
        event_handler=_event_handler,
        feeding=feeding,
        num_passes=conf.num_passes)

    logger.info("Training has finished.")


if __name__ == "__main__":
    train()
