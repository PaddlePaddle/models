#coding=utf-8

from __future__ import print_function

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


def load_initial_model(model_path, parameters):
    """ Initalize parameters in the network from a trained model.

    This is useful in resuming the training from previously saved models.

    Arguments:
        - model_path:    The path of a trained model.
        - parameters:    The parameters in a network which will be initialized
                         from the specified model.
    """
    with gzip.open(model_path, "rb") as f:
        parameters.init_from_tar(f)


def load_pretrained_parameters(path):
    """ Load one pre-trained parameter.

    Arguments:
        - path:    The path of the pre-trained parameter.
    """
    return np.load(path)


def save_model(trainer, save_path, parameters):
    """ Save the trained parameters.

    Arguments:
        - save_path:    The path to save the trained parameters.
        - parameters:   The trained model parameters.
    """
    with gzip.open(save_path, "w") as f:
        trainer.save_parameter_to_tar(f)


def show_parameter_init_info(parameters):
    """ Print the information of initialization mean and std of parameters.

    Arguments:
        - parameters:   The parameters created in a model.
    """
    for p in parameters:
        logger.info("%s : initial_mean %.4f initial_std %.4f" %
                    (p, parameters.__param_conf__[p].initial_mean,
                     parameters.__param_conf__[p].initial_std))


def show_parameter_status(parameters):
    """ Print some statistical information of parameters in a network.

    This is used for debugging the model.

    Arguments:
        - parameters:   The parameters created in a model.
    """
    for p in parameters:

        value = parameters.get(p)
        grad = parameters.get_grad(p)

        avg_abs_value = np.average(np.abs(value))
        avg_abs_grad = np.average(np.abs(grad))

        logger.info(
            ("%s avg_abs_value=%.6f avg_abs_grad=%.6f "
             "min_value=%.6f max_value=%.6f min_grad=%.6f max_grad=%.6f") %
            (p, avg_abs_value, avg_abs_grad, value.min(), value.max(),
             grad.min(), grad.max()))


def choose_samples(path):
    """Load filenames for train, dev, and augmented samples.

    Arguments:
        - path:   The path of training data.
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


def build_reader(data_dir, batch_size):
    """Build the data reader for this model.

    Arguments:
        - data_dir:   The path of training data.
        - batch_size:   batch size for the training task.
    """
    train_samples, valid_samples = choose_samples(data_dir)

    train_reader = paddle.batch(
        paddle.reader.shuffle(
            reader.data_reader(train_samples), buf_size=102400),
        batch_size=batch_size)

    # testing data is not shuffled
    test_reader = paddle.batch(
        reader.data_reader(
            valid_samples, is_train=False),
        batch_size=batch_size)
    return train_reader, test_reader, len(train_samples)


def build_event_handler(config, parameters, trainer):
    """Build the event handler for this model.

    Arguments:
        - config:        The training task configuration for this model.
        - parameters:    The parameters in the network.
        - trainer:       The trainer object.
    """

    # End batch and end pass event handler
    def event_handler(event):
        """The event handler."""
        """
        To print the statistical information of gradients of any learnable
        parameter, the event: EndForwardBackward rather than EndIteration
        should be handled. For the reason that parameter gradients will be
        reset to zeros when EndIteration event happens in GPU training.
        """
        if config.show_parameter_status_period and \
                isinstance(event, paddle.event.EndForwardBackward):
            if not event.batch_id % config.show_parameter_status_period:
                show_parameter_status(parameters)

        if isinstance(event, paddle.event.EndIteration):
            if event.batch_id and not event.batch_id % config.checkpoint_period:
                save_path = os.path.join(config.save_dir,
                                         "checkpoint_param.latest.tar.gz")
                save_model(trainer, save_path, parameters)

            if not event.batch_id % config.log_period:
                logger.info("Pass %d, Batch %d, Cost %f" %
                            (event.pass_id, event.batch_id, event.cost))

        if isinstance(event, paddle.event.EndPass):
            save_path = os.path.join(config.save_dir,
                                     "pass_%05d.tar.gz" % event.pass_id)
            save_model(trainer, save_path, parameters)

    return event_handler


def train(model_config, trainer_config):
    """Training the GNR model.

    Arguments:
        - modle_config:     The model configuration for this model.
        - trainer_config:   The training task configuration for this model.
    """

    if not os.path.exists(trainer_config.save_dir):
        os.mkdir(trainer_config.save_dir)

    paddle.init(
        use_gpu=trainer_config.use_gpu,
        trainer_count=trainer_config.trainer_count)

    train_reader, test_reader, train_sample_count = build_reader(
        trainer_config.data_dir, trainer_config.train_batch_size)
    """
    Define the optimizer. The learning rate will decrease according to
    the following formula:

    lr = learning_rate * pow(learning_rate_decay_a,
                             floor(num_samples_processed /
                                   learning_rate_decay_b))
    """
    optimizer = paddle.optimizer.Adam(
        learning_rate=trainer_config.learning_rate,
        gradient_clipping_threshold=trainer_config.gradient_clipping_threshold,
        regularization=paddle.optimizer.L2Regularization(
            trainer_config.l2_decay_rate),
        learning_rate_decay_a=0.5,
        learning_rate_decay_b=train_sample_count,
        learning_rate_schedule="discexp")

    # define network topology
    loss = GNR(model_config)

    parameters = paddle.parameters.create(loss)

    if trainer_config.init_model_path:
        load_initial_model(trainer_config.init_model_path, parameters)
    else:
        show_parameter_init_info(parameters)
        parameters.set(
            "GloveVectors",
            load_pretrained_parameters(ModelConfig.pretrained_emb_path))

    trainer = paddle.trainer.SGD(cost=loss,
                                 parameters=parameters,
                                 update_equation=optimizer)

    event_handler = build_event_handler(trainer_config, parameters, trainer)
    trainer.train(
        reader=train_reader,
        num_passes=trainer_config.epochs,
        event_handler=event_handler)


if __name__ == "__main__":
    train(ModelConfig, TrainerConfig)
