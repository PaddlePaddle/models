import os
import sys
import gzip

import paddle.v2 as paddle

import reader
from utils import logger, parse_train_cmd, build_dict, load_dict
from network_conf import fc_net, convolution_net


def train(topology,
          train_data_dir=None,
          test_data_dir=None,
          word_dict_path=None,
          label_dict_path=None,
          model_save_dir="models",
          batch_size=32,
          num_passes=10):
    """
    train dnn model


    :params train_data_path: path of training data, if this parameter
        is not specified, paddle.dataset.imdb will be used to run this example
    :type train_data_path: str
    :params test_data_path: path of testing data, if this parameter
        is not specified, paddle.dataset.imdb will be used to run this example
    :type test_data_path: str
    :params word_dict_path: path of training data, if this parameter
        is not specified, paddle.dataset.imdb will be used to run this example
    :type word_dict_path: str
    :params num_pass: train pass number
    :type num_pass: int
    """
    if not os.path.exists(model_save_dir):
        os.mkdir(model_save_dir)

    use_default_data = (train_data_dir is None)

    if use_default_data:
        logger.info(("No training data are provided, "
                     "use paddle.dataset.imdb to train the model."))
        logger.info("please wait to build the word dictionary ...")

        word_dict = paddle.dataset.imdb.word_dict()
        train_reader = paddle.batch(
            paddle.reader.shuffle(
                lambda: paddle.dataset.imdb.train(word_dict)(), buf_size=51200),
            batch_size=100)
        test_reader = paddle.batch(
            lambda: paddle.dataset.imdb.test(word_dict)(), batch_size=100)

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
                cutoff_fre=5,
                insert_extra_words=["<UNK>"])

        if not os.path.exists(label_dict_path):
            logger.info(("label dictionary is not given, the dictionary "
                         "is automatically built from the training data."))
            # build the label dictionary to map the original string-typed
            # label into integer-typed index
            build_dict(
                data_dir=train_data_dir, save_path=label_dict_path, use_col=0)

        word_dict = load_dict(word_dict_path)

        lbl_dict = load_dict(label_dict_path)
        class_num = len(lbl_dict)
        logger.info("class number is : %d." % (len(lbl_dict)))

        train_reader = paddle.batch(
            paddle.reader.shuffle(
                reader.train_reader(train_data_dir, word_dict, lbl_dict),
                buf_size=51200),
            batch_size=batch_size)

        if test_data_dir is not None:
            # here, because training and testing data share a same format,
            # we still use the reader.train_reader to read the testing data.
            test_reader = paddle.batch(
                reader.train_reader(test_data_dir, word_dict, lbl_dict),
                batch_size=batch_size)
        else:
            test_reader = None

    dict_dim = len(word_dict)
    logger.info("length of word dictionary is : %d." % (dict_dim))

    paddle.init(use_gpu=False, trainer_count=1)

    # network config
    cost, prob, label = topology(dict_dim, class_num)

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
                    os.path.join(model_save_dir, "dnn_params_pass_%05d.tar.gz" %
                                 event.pass_id), "w") as f:
                trainer.save_parameter_to_tar(f)

    trainer.train(
        reader=train_reader,
        event_handler=_event_handler,
        feeding=feeding,
        num_passes=num_passes)

    logger.info("Training has finished.")


def main(args):
    if args.nn_type == "dnn":
        topology = fc_net
    elif args.nn_type == "cnn":
        topology = convolution_net

    train(
        topology=topology,
        train_data_dir=args.train_data_dir,
        test_data_dir=args.test_data_dir,
        word_dict_path=args.word_dict,
        label_dict_path=args.label_dict,
        batch_size=args.batch_size,
        num_passes=args.num_passes,
        model_save_dir=args.model_save_dir)


if __name__ == "__main__":
    args = parse_train_cmd()
    if args.train_data_dir is not None:
        assert args.word_dict and args.label_dict, (
            "the parameter train_data_dir, word_dict_path, and label_dict_path "
            "should be set at the same time.")
    main(args)
