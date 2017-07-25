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

import reader
import paddle.v2 as paddle
from paddle.v2.layer import parse_network
from model import GNR
from config import ModelConfig, TrainerConfig

logger = logging.getLogger("paddle")
logger.setLevel(logging.INFO)


def load_pretrained_parameters(path, height, width):
    return


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

    random.shuffle(train_samples)

    return train_samples, valid_samples


def build_reader(data_dir):
    """
    Build the data reader for this model.
    """
    train_samples, valid_samples = choose_samples(data_dir)
    pdb.set_trace()

    train_reader = paddle.batch(
        paddle.reader.shuffle(
            reader.train_reader(train_samples), buf_size=102400),
        batch_size=config.batch_size)

    # testing data is not shuffled
    test_reader = paddle.batch(
        reader.train_reader(valid_samples, is_train=False),
        batch_size=config.batch_size)
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
                save_model("checkpoint_param.latest.tar.gz", parameters)

            if not event.batch_id % 5:
                logger.info(
                    "Pass %d, Batch %d, Cost %f, %s" %
                    (event.pass_id, event.batch_id, event.cost, event.metrics))

        if isinstance(event, paddle.event.EndPass):
            save_model(config.param_save_filename_format % event.pass_id,
                       parameters)
            with gzip.open(param_path, 'w') as handle:
                parameters.to_tar(handle)

            result = trainer.test(reader=test_reader)
            logger.info("Test with Pass %d, %s" %
                        (event.pass_id, result.metrics))

    return event_handler


def train(model_config, trainer_config):
    paddle.init(use_gpu=True, trainer_count=1)

    # define the optimizer
    optimizer = paddle.optimizer.Adam(
        learning_rate=trainer_config.learning_rate,
        regularization=paddle.optimizer.L2Regularization(rate=1e-3),
        model_average=paddle.optimizer.ModelAverage(average_window=0.5))

    # define network topology
    losses = GNR(model_config)
    parameters = paddle.parameters.create(losses)
    # print(parse_network(losses))
    trainer = paddle.trainer.SGD(
        cost=losses, parameters=parameters, update_equation=optimizer)
    """
    parameters.set('GloveVectors',
                   load_pretrained_parameters(parameter_path, height, width))
    """

    # define data reader
    train_reader, test_reader = build_reader(trainer_config.data_dir)

    event_handler = build_event_handler(conf, parameters, trainer, test_reader)
    trainer.train(
        reader=train_reader,
        num_passes=conf.epochs,
        event_handler=event_handler)


if __name__ == "__main__":
    train(ModelConfig, TrainerConfig)
