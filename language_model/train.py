#!/usr/bin/env python
# coding=utf-8
import os
import sys
import gzip
import pdb

import paddle.v2 as paddle
import config as conf
import network_conf
import reader
from utils import logger, build_dict, load_dict


def train(model_cost, train_reader, test_reader, save_prefix, num_passes):
    """
    train model.

    :param model_cost: cost layer of the model to train.
    :param train_reader: train data reader.
    :param test_reader: test data reader.
    :param model_file_name_prefix: model"s prefix name.
    :param num_passes: epoch.
    :return:
    """

    paddle.init(use_gpu=conf.use_gpu, trainer_count=conf.trainer_count)

    # create parameters
    parameters = paddle.parameters.create(model_cost)

    # create optimizer
    adam_optimizer = paddle.optimizer.Adam(
        learning_rate=1e-3,
        regularization=paddle.optimizer.L2Regularization(rate=1e-3),
        model_average=paddle.optimizer.ModelAverage(
            average_window=0.5, max_average_window=10000))

    # create trainer
    trainer = paddle.trainer.SGD(
        cost=model_cost, parameters=parameters, update_equation=adam_optimizer)

    # define event_handler callback
    def event_handler(event):
        if isinstance(event, paddle.event.EndIteration):
            if event.batch_id % 1 == 0:
                logger.info("Pass %d, Batch %d, Cost %f, %s" % (
                    event.pass_id, event.batch_id, event.cost, event.metrics))
            if not (event.batch_id + 1 % 200):
                with gzip.open("models/%s_batch_%05d.tar.gz" %
                               (save_prefix, event.batch_id), "w") as f:
                    parameters.to_tar(f)

        # save model each pass
        if isinstance(event, paddle.event.EndPass):
            if test_reader is not None:
                result = trainer.test(reader=test_reader)
                logger.info("Test with Pass %d, %s" %
                            (event.pass_id, result.metrics))
            with gzip.open("models/%s_pass_%05d.tar.gz" %
                           (save_prefix, event.pass_id), "w") as f:
                parameters.to_tar(f)

    # start to train
    logger.info("start training...")
    trainer.train(
        reader=train_reader, event_handler=event_handler, num_passes=num_passes)

    logger.info("Training is finished.")


def main():
    # prepare vocab
    if not (os.path.exists(conf.vocab_file) and
            os.path.getsize(conf.vocab_file)):
        logger.info(("word dictionary does not exist, "
                     "build it from the training data"))
        build_dict(
            data_file=conf.train_file,
            save_path=conf.vocab_file,
            max_word_num=conf.max_word_num)
    logger.info("load word dictionary.")
    word_dict = load_dict(conf.vocab_file)
    logger.info("dictionay size = %d" % (len(word_dict)))

    reader_func = None
    args = {
        "file_name": conf.train_file,
        "word_dict": word_dict,
        "is_infer": False
    }
    if conf.model_type == "rnn":
        logger.info("prepare rnn model...")
        rnn_conf = conf.ConfigRnn()
        cost = network_conf.rnn_lm(
            len(word_dict), rnn_conf.emb_dim, rnn_conf.rnn_type,
            rnn_conf.hidden_size, rnn_conf.num_layer)
        reader_func = reader.rnn_reader

    elif conf.model_type == "ngram":
        logger.info("prepare ngram model...")
        ngram_conf = conf.ConfigNgram()
        cost = network_conf.ngram_lm(
            vocab_size=len(word_dict),
            emb_dim=ngram_conf.emb_dim,
            hidden_size=ngram_conf.hidden_size,
            num_layer=ngram_conf.num_layer,
            gram_num=ngram_conf.N)

        reader_func = reader.ngram_reader
        args["gram_num"] = ngram_conf.N

    else:
        raise Exception("wrong model type [currently supported: rnn / ngrm]")

    train_reader = paddle.batch(
        paddle.reader.shuffle(reader_func(**args), buf_size=65536),
        batch_size=conf.batch_size)

    test_reader = None
    if os.path.exists(conf.test_file) and os.path.getsize(conf.test_file):
        test_reader = paddle.batch(
            paddle.reader.shuffle(reader_func(**args), buf_size=65536),
            batch_size=config.batch_size)

    # train model
    train(
        model_cost=cost,
        train_reader=train_reader,
        test_reader=test_reader,
        save_prefix=conf.model_type,
        num_passes=conf.num_passes)


if __name__ == "__main__":
    main()
