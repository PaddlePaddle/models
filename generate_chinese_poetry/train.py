#!/usr/bin/env python
#coding=utf-8

import gzip
import os
import logging

import paddle.v2 as paddle
import reader
from paddle.v2.layer import parse_network
from network_conf import encoder_decoder_network

logger = logging.getLogger("paddle")
logger.setLevel(logging.INFO)


def save_model(save_path, parameters):
    with gzip.open(save_path, "w") as f:
        parameters.to_tar(f)


def load_initial_model(model_path, parameters):
    with gzip.open(model_path, "rb") as f:
        parameters.init_from_tar(f)


def main(num_passes,
         batch_size,
         use_gpu,
         trainer_count,
         save_dir_path,
         encoder_depth,
         decoder_depth,
         word_dict_path,
         train_data_path,
         init_model_path=""):
    if not os.path.exists(save_dir_path):
        os.mkdir(save_dir_path)

    # initialize PaddlePaddle
    paddle.init(use_gpu=use_gpu, trainer_count=trainer_count, parallel_nn=1)

    # define optimization method and the trainer instance
    # optimizer = paddle.optimizer.Adam(
    optimizer = paddle.optimizer.AdaDelta(
        learning_rate=1e-3,
        gradient_clipping_threshold=25.0,
        regularization=paddle.optimizer.L2Regularization(rate=8e-4),
        model_average=paddle.optimizer.ModelAverage(
            average_window=0.5, max_average_window=2500))

    cost = encoder_decoder_network(
        word_count=len(open(word_dict_path, "r").readlines()),
        emb_dim=512,
        encoder_depth=encoder_depth,
        encoder_hidden_dim=512,
        decoder_depth=decoder_depth,
        decoder_hidden_dim=512)

    parameters = paddle.parameters.create(cost)
    if init_model_path:
        load_initial_model(init_model_path, parameters)

    trainer = paddle.trainer.SGD(
        cost=cost, parameters=parameters, update_equation=optimizer)

    # define data reader
    train_reader = paddle.batch(
        paddle.reader.shuffle(
            reader.train_reader(train_data_path, word_dict_path),
            buf_size=1024000),
        batch_size=batch_size)

    # define the event_handler callback
    def event_handler(event):
        if isinstance(event, paddle.event.EndIteration):
            if (not event.batch_id % 2000) and event.batch_id:
                save_path = os.path.join(save_dir_path,
                                         "pass_%05d_batch_%05d.tar.gz" %
                                         (event.pass_id, event.batch_id))
                save_model(save_path, parameters)

            if not event.batch_id % 5:
                logger.info("Pass %d, Batch %d, Cost %f, %s" % (
                    event.pass_id, event.batch_id, event.cost, event.metrics))

        if isinstance(event, paddle.event.EndPass):
            save_path = os.path.join(save_dir_path,
                                     "pass_%05d.tar.gz" % event.pass_id)
            save_model(save_path, parameters)

    # start training
    trainer.train(
        reader=train_reader, event_handler=event_handler, num_passes=num_passes)


if __name__ == '__main__':
    main(
        num_passes=500,
        batch_size=4 * 500,
        use_gpu=True,
        trainer_count=4,
        encoder_depth=3,
        decoder_depth=3,
        save_dir_path="models",
        word_dict_path="data/word_dict.txt",
        train_data_path="data/song.poet.txt",
        init_model_path="")
