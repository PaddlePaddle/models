import os
import gzip
import logging
import click

import paddle.v2 as paddle
import reader
from paddle.v2.layer import parse_network
from network_conf import encoder_decoder_network

logger = logging.getLogger("paddle")
logger.setLevel(logging.INFO)


def save_model(trainer, save_path, parameters):
    with gzip.open(save_path, "w") as f:
        trainer.save_parameter_to_tar(f)


def load_initial_model(model_path, parameters):
    with gzip.open(model_path, "rb") as f:
        parameters.init_from_tar(f)


@click.command("train")
@click.option(
    "--num_passes", default=10, help="Number of passes for the training task.")
@click.option(
    "--batch_size",
    default=16,
    help="The number of training examples in one forward/backward pass.")
@click.option(
    "--use_gpu", default=False, help="Whether to use gpu to train the model.")
@click.option(
    "--trainer_count", default=1, help="The thread number used in training.")
@click.option(
    "--save_dir_path",
    default="models",
    help="The path to saved the trained models.")
@click.option(
    "--encoder_depth",
    default=3,
    help="The number of stacked LSTM layers in encoder.")
@click.option(
    "--decoder_depth",
    default=3,
    help="The number of stacked LSTM layers in decoder.")
@click.option(
    "--train_data_path", required=True, help="The path of trainning data.")
@click.option(
    "--word_dict_path", required=True, help="The path of word dictionary.")
@click.option(
    "--init_model_path",
    default="",
    help=("The path of a trained model used to initialized all "
          "the model parameters."))
def train(num_passes,
          batch_size,
          use_gpu,
          trainer_count,
          save_dir_path,
          encoder_depth,
          decoder_depth,
          train_data_path,
          word_dict_path,
          init_model_path=""):
    if not os.path.exists(save_dir_path):
        os.mkdir(save_dir_path)
    assert os.path.exists(
        word_dict_path), "The given word dictionary does not exist."
    assert os.path.exists(
        train_data_path), "The given training data does not exist."

    # initialize PaddlePaddle
    paddle.init(use_gpu=use_gpu, trainer_count=trainer_count)

    # define optimization method and the trainer instance
    optimizer = paddle.optimizer.Adam(
        learning_rate=1e-4,
        regularization=paddle.optimizer.L2Regularization(rate=1e-5),
        model_average=paddle.optimizer.ModelAverage(
            average_window=0.5, max_average_window=2500))

    cost = encoder_decoder_network(
        word_count=len(open(word_dict_path, "r").readlines()),
        emb_dim=512,
        encoder_depth=encoder_depth,
        encoder_hidden_dim=512,
        decoder_depth=decoder_depth,
        decoder_hidden_dim=512,
        bos_id=0,
        eos_id=1,
        max_length=9)

    parameters = paddle.parameters.create(cost)
    if init_model_path:
        load_initial_model(init_model_path, parameters)

    trainer = paddle.trainer.SGD(cost=cost,
                                 parameters=parameters,
                                 update_equation=optimizer)

    # define data reader
    train_reader = paddle.batch(
        paddle.reader.shuffle(
            reader.train_reader(train_data_path, word_dict_path),
            buf_size=1024000),
        batch_size=batch_size)

    # define the event_handler callback
    def event_handler(event):
        if isinstance(event, paddle.event.EndIteration):
            if (not event.batch_id % 1000) and event.batch_id:
                save_path = os.path.join(save_dir_path,
                                         "pass_%05d_batch_%05d.tar.gz" %
                                         (event.pass_id, event.batch_id))
                save_model(trainer, save_path, parameters)

            if not event.batch_id % 10:
                logger.info("Pass %d, Batch %d, Cost %f, %s" % (
                    event.pass_id, event.batch_id, event.cost, event.metrics))

        if isinstance(event, paddle.event.EndPass):
            save_path = os.path.join(save_dir_path,
                                     "pass_%05d.tar.gz" % event.pass_id)
            save_model(trainer, save_path, parameters)

    # start training
    trainer.train(
        reader=train_reader, event_handler=event_handler, num_passes=num_passes)


if __name__ == "__main__":
    train()
