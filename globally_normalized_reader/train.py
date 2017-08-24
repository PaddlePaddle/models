#!/usr/bin/env python
#coding=utf-8
from __future__ import print_function

import pdb
import os
import sys
import logging
import random
import glob
import gzip
import numpy as np

import reader
import paddle.v2 as paddle
from paddle.v2.layer import parse_network
from model import GNR
from config import ModelConfig, TrainerConfig

logger = logging.getLogger("paddle")
logger.setLevel(logging.INFO)


def load_pretrained_parameters(path, height, width):
    return np.load(path)


def save_model(save_path, parameters):
    with gzip.open(save_path, "w") as f:
        parameters.to_tar(f)


def load_initial_model(model_path, parameters):
    with gzip.open(model_path, "rb") as f:
        parameters.init_from_tar(f)


def choose_samples(path):
    """
    Load filenames for train, dev, and augmented samples.
    """
    if not os.path.exists(os.path.join(path, "train")):
        print(
            "Non-existent directory as input path: {}".format(path),
            file=sys.stderr)
        sys.exit(1)

    # Get paths to all samples that we want to load.
    train_samples = glob.glob(os.path.join(path, "train", "*"))
    valid_samples = glob.glob(os.path.join(path, "dev", "*"))

    train_samples.sort()
    valid_samples.sort()

    # random.shuffle(train_samples)

    return train_samples, valid_samples


def build_reader(data_dir, batch_size):
    """
    Build the data reader for this model.
    """
    train_samples, valid_samples = choose_samples(data_dir)

    train_reader = paddle.batch(
        paddle.reader.shuffle(
            reader.train_reader(train_samples), buf_size=102400),
        batch_size=batch_size)

    # train_reader = paddle.batch(
    #     reader.train_reader(train_samples), batch_size=batch_size)

    # testing data is not shuffled
    test_reader = paddle.batch(
        reader.train_reader(
            valid_samples, is_train=False),
        batch_size=batch_size)
    return train_reader, test_reader


def build_event_handler(config, parameters, trainer, test_reader):
    """
    Build the event handler for this model.
    """

    # End batch and end pass event handler
    def event_handler(event):
        """The event handler."""
        if isinstance(event, paddle.event.EndIteration):
            if (not event.batch_id % 100) and event.batch_id:
                save_path = os.path.join(config.save_dir,
                                         "checkpoint_param.latest.tar.gz")
                save_model(save_path, parameters)

            if not event.batch_id % 1:
                logger.info(
                    "Pass %d, Batch %d, Cost %f, %s" %
                    (event.pass_id, event.batch_id, event.cost, event.metrics))

        if isinstance(event, paddle.event.EndPass):
            save_path = os.path.join(config.save_dir,
                                     "pass_%05d.tar.gz" % event.pass_id)
            save_model(save_path, parameters)

            # result = trainer.test(reader=test_reader)
            # logger.info("Test with Pass %d, %s" %
            #             (event.pass_id, result.metrics))

    return event_handler


def train(model_config, trainer_config):
    if not os.path.exists(trainer_config.save_dir):
        os.mkdir(trainer_config.save_dir)

    paddle.init(use_gpu=True, trainer_count=4)

    # define the optimizer
    optimizer = paddle.optimizer.Adam(
        learning_rate=trainer_config.learning_rate,
        regularization=paddle.optimizer.L2Regularization(rate=1e-3),
        # model_average=paddle.optimizer.ModelAverage(average_window=0.5))
    )

    # define network topology
    loss = GNR(model_config)

    # print(parse_network(loss))

    parameters = paddle.parameters.create(loss)
    parameters.set("GloveVectors",
                   load_pretrained_parameters(
                       ModelConfig.pretrained_emb_path,
                       height=ModelConfig.vocab_size,
                       width=ModelConfig.embedding_dim))

    trainer = paddle.trainer.SGD(cost=loss,
                                 parameters=parameters,
                                 update_equation=optimizer)

    # define data reader
    train_reader, test_reader = build_reader(trainer_config.data_dir,
                                             trainer_config.batch_size)

    event_handler = build_event_handler(trainer_config, parameters, trainer,
                                        test_reader)
    trainer.train(
        reader=data_reader,
        num_passes=trainer_config.epochs,
        event_handler=event_handler)


if __name__ == "__main__":
    train(ModelConfig, TrainerConfig)
