"""
    Contains training script for machine translation with external memory.
"""
import argparse
import sys
import gzip
import distutils.util
import random

import paddle.v2 as paddle
from external_memory import ExternalMemory
from model import memory_enhanced_seq2seq
from data_utils import reader_append_wrapper

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--dict_size",
    default=30000,
    type=int,
    help="Vocabulary size. (default: %(default)s)")
parser.add_argument(
    "--word_vec_dim",
    default=512,
    type=int,
    help="Word embedding size. (default: %(default)s)")
parser.add_argument(
    "--hidden_size",
    default=1024,
    type=int,
    help="Hidden cell number in RNN. (default: %(default)s)")
parser.add_argument(
    "--memory_slot_num",
    default=8,
    type=int,
    help="External memory slot number. (default: %(default)s)")
parser.add_argument(
    "--use_gpu",
    default=False,
    type=distutils.util.strtobool,
    help="Use gpu or not. (default: %(default)s)")
parser.add_argument(
    "--trainer_count",
    default=1,
    type=int,
    help="Trainer number. (default: %(default)s)")
parser.add_argument(
    "--num_passes",
    default=100,
    type=int,
    help="Training epochs. (default: %(default)s)")
parser.add_argument(
    "--batch_size",
    default=5,
    type=int,
    help="Batch size. (default: %(default)s)")
parser.add_argument(
    "--memory_perturb_stddev",
    default=0.1,
    type=float,
    help="Memory perturb stddev for memory initialization."
    "(default: %(default)s)")
args = parser.parse_args()


def train():
    """
    For training.
    """
    # create optimizer
    optimizer = paddle.optimizer.Adam(
        learning_rate=5e-5,
        gradient_clipping_threshold=5,
        regularization=paddle.optimizer.L2Regularization(rate=8e-4))

    # create network config
    source_words = paddle.layer.data(
        name="source_words",
        type=paddle.data_type.integer_value_sequence(args.dict_size))
    target_words = paddle.layer.data(
        name="target_words",
        type=paddle.data_type.integer_value_sequence(args.dict_size))
    target_next_words = paddle.layer.data(
        name='target_next_words',
        type=paddle.data_type.integer_value_sequence(args.dict_size))
    cost = memory_enhanced_seq2seq(
        encoder_input=source_words,
        decoder_input=target_words,
        decoder_target=target_next_words,
        hidden_size=args.hidden_size,
        word_vec_dim=args.word_vec_dim,
        dict_size=args.dict_size,
        is_generating=False,
        beam_size=None)

    # create parameters and trainer
    parameters = paddle.parameters.create(cost)
    trainer = paddle.trainer.SGD(cost=cost,
                                 parameters=parameters,
                                 update_equation=optimizer)

    # create data readers
    feeding = {
        "source_words": 0,
        "target_words": 1,
        "target_next_words": 2,
        "bounded_memory_perturbation": 3
    }
    random.seed(0)  # for keeping consitancy for multiple runs
    bounded_memory_perturbation = [[
        random.gauss(0, args.memory_perturb_stddev)
        for i in xrange(args.hidden_size)
    ] for j in xrange(args.memory_slot_num)]
    train_append_reader = reader_append_wrapper(
        reader=paddle.dataset.wmt14.train(args.dict_size),
        append_tuple=(bounded_memory_perturbation, ))
    train_batch_reader = paddle.batch(
        reader=paddle.reader.shuffle(
            reader=train_append_reader, buf_size=8192),
        batch_size=args.batch_size)
    test_append_reader = reader_append_wrapper(
        reader=paddle.dataset.wmt14.test(args.dict_size),
        append_tuple=(bounded_memory_perturbation, ))
    test_batch_reader = paddle.batch(
        reader=paddle.reader.shuffle(
            reader=test_append_reader, buf_size=8192),
        batch_size=args.batch_size)

    # create event handler
    def event_handler(event):
        if isinstance(event, paddle.event.EndIteration):
            if event.batch_id % 10 == 0:
                print "Pass: %d, Batch: %d, TrainCost: %f, %s" % (
                    event.pass_id, event.batch_id, event.cost, event.metrics)
                with gzip.open("checkpoints/params.latest.tar.gz", 'w') as f:
                    trainer.save_parameter_to_tar(f)
            else:
                sys.stdout.write('.')
                sys.stdout.flush()
        if isinstance(event, paddle.event.EndPass):
            result = trainer.test(reader=test_batch_reader, feeding=feeding)
            print "Pass: %d, TestCost: %f, %s" % (event.pass_id, result.cost,
                                                  result.metrics)
            with gzip.open("checkpoints/params.pass-%d.tar.gz" % event.pass_id,
                           'w') as f:
                trainer.save_parameter_to_tar(f)

    # run train
    if not os.path.exists('checkpoints'):
        os.mkdir('checkpoints')
    trainer.train(
        reader=train_batch_reader,
        event_handler=event_handler,
        num_passes=args.num_passes,
        feeding=feeding)


def main():
    paddle.init(use_gpu=args.use_gpu, trainer_count=args.trainer_count)
    train()


if __name__ == '__main__':
    main()
