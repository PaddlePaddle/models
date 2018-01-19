import os
import sys
import gzip

import paddle.v2 as paddle
import config as conf
import reader
from network_conf import rnn_lm
from utils import logger, build_dict, load_dict


def train(topology,
          train_reader,
          test_reader,
          model_save_dir="models",
          num_passes=10):
    """
    train model.

    :param topology: cost layer of the model to train.
    :type topology: LayerOuput
    :param train_reader: train data reader.
    :type trainer_reader: collections.Iterable
    :param test_reader: test data reader.
    :type test_reader: collections.Iterable
    :param model_save_dir: path to save the trained model
    :type model_save_dir: str
    :param num_passes: number of epoch
    :type num_passes: int
    """
    if not os.path.exists(model_save_dir):
        os.mkdir(model_save_dir)

    # initialize PaddlePaddle
    paddle.init(use_gpu=conf.use_gpu, trainer_count=conf.trainer_count)

    # create optimizer
    adam_optimizer = paddle.optimizer.Adam(
        learning_rate=1e-3,
        regularization=paddle.optimizer.L2Regularization(rate=1e-3),
        model_average=paddle.optimizer.ModelAverage(
            average_window=0.5, max_average_window=10000))

    # create parameters
    parameters = paddle.parameters.create(topology)
    # create sum evaluator
    sum_eval = paddle.evaluator.sum(topology)
    # create trainer
    trainer = paddle.trainer.SGD(cost=topology,
                                 parameters=parameters,
                                 update_equation=adam_optimizer,
                                 extra_layers=sum_eval)

    # define the event_handler callback
    def event_handler(event):
        if isinstance(event, paddle.event.EndIteration):
            if not event.batch_id % conf.log_period:
                logger.info("Pass %d, Batch %d, Cost %f, %s" % (
                    event.pass_id, event.batch_id, event.cost, event.metrics))

            if (not event.batch_id %
                    conf.save_period_by_batches) and event.batch_id:
                save_name = os.path.join(model_save_dir,
                                         "rnn_lm_pass_%05d_batch_%03d.tar.gz" %
                                         (event.pass_id, event.batch_id))
                with gzip.open(save_name, "w") as f:
                    trainer.save_parameter_to_tar(f)

        if isinstance(event, paddle.event.EndPass):
            if test_reader is not None:
                result = trainer.test(reader=test_reader)
                logger.info("Test with Pass %d, %s" %
                            (event.pass_id, result.metrics))
            save_name = os.path.join(model_save_dir, "rnn_lm_pass_%05d.tar.gz" %
                                     (event.pass_id))
            with gzip.open(save_name, "w") as f:
                trainer.save_parameter_to_tar(f)

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
        build_dict(conf.train_file, conf.vocab_file, conf.max_word_num,
                   conf.cutoff_word_fre)
    logger.info("load word dictionary.")
    word_dict = load_dict(conf.vocab_file)
    logger.info("dictionay size = %d" % (len(word_dict)))

    cost = rnn_lm(
        len(word_dict), conf.emb_dim, conf.hidden_size, conf.stacked_rnn_num,
        conf.rnn_type)

    # define reader
    reader_args = {
        "file_name": conf.train_file,
        "word_dict": word_dict,
    }
    train_reader = paddle.batch(
        paddle.reader.shuffle(
            reader.rnn_reader(**reader_args), buf_size=102400),
        batch_size=conf.batch_size)
    test_reader = None
    if os.path.exists(conf.test_file) and os.path.getsize(conf.test_file):
        test_reader = paddle.batch(
            paddle.reader.shuffle(
                reader.rnn_reader(**reader_args), buf_size=65536),
            batch_size=conf.batch_size)

    train(
        topology=cost,
        train_reader=train_reader,
        test_reader=test_reader,
        model_save_dir=conf.model_save_dir,
        num_passes=conf.num_passes)


if __name__ == "__main__":
    main()
