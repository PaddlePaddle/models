import os
import sys
import gzip
import argparse
import distutils.util
import paddle.v2 as paddle

import reader
from network_conf import seqToseq_net


def parse_args():
    parser = argparse.ArgumentParser(
        description="PaddlePaddle Scheduled Sampling")
    parser.add_argument(
        '--schedule_type',
        type=str,
        default="linear",
        help='The type of sampling rate decay. Supported type: constant, linear, exponential, inverse_sigmoid. (default: %(default)s)'
    )
    parser.add_argument(
        '--decay_a',
        type=float,
        default=0.75,
        help='The sampling rate decay parameter a. (default: %(default)s)')
    parser.add_argument(
        '--decay_b',
        type=float,
        default=1000000,
        help='The sampling rate decay parameter b. (default: %(default)s)')
    parser.add_argument(
        '--beam_size',
        type=int,
        default=3,
        help='The width of beam expansion. (default: %(default)s)')
    parser.add_argument(
        "--use_gpu",
        type=distutils.util.strtobool,
        default=False,
        help="Use gpu or not. (default: %(default)s)")
    parser.add_argument(
        "--trainer_count",
        type=int,
        default=1,
        help="Trainer number. (default: %(default)s)")
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help="Size of a mini-batch. (default: %(default)s)")
    parser.add_argument(
        '--num_passes',
        type=int,
        default=10,
        help="Number of passes to train. (default: %(default)s)")
    parser.add_argument(
        '--model_output_dir',
        type=str,
        default='models',
        help="The path for model to store. (default: %(default)s)")

    return parser.parse_args()


def train(dict_size, batch_size, num_passes, beam_size, schedule_type, decay_a,
          decay_b, model_dir):
    optimizer = paddle.optimizer.Adam(
        learning_rate=1e-4,
        regularization=paddle.optimizer.L2Regularization(rate=1e-5))

    cost = seqToseq_net(dict_size, dict_size, beam_size)

    parameters = paddle.parameters.create(cost)

    trainer = paddle.trainer.SGD(cost=cost,
                                 parameters=parameters,
                                 update_equation=optimizer)

    wmt14_reader = reader.gen_schedule_data(
        paddle.reader.shuffle(
            paddle.dataset.wmt14.train(dict_size), buf_size=8192),
        schedule_type,
        decay_a,
        decay_b)

    # define event_handler callback
    def event_handler(event):
        if isinstance(event, paddle.event.EndIteration):
            if event.batch_id % 10 == 0:
                print "\nPass %d, Batch %d, Cost %f, %s" % (
                    event.pass_id, event.batch_id, event.cost, event.metrics)
            else:
                sys.stdout.write('.')
                sys.stdout.flush()
        if isinstance(event, paddle.event.EndPass):
            # save parameters
            with gzip.open(
                    os.path.join(model_dir, 'params_pass_%d.tar.gz' %
                                 event.pass_id), 'w') as f:
                trainer.save_parameter_to_tar(f)

    # start to train
    trainer.train(
        reader=paddle.batch(
            wmt14_reader, batch_size=batch_size),
        event_handler=event_handler,
        feeding=reader.feeding,
        num_passes=num_passes)


if __name__ == '__main__':
    args = parse_args()

    if not os.path.isdir(args.model_output_dir):
        os.mkdir(args.model_output_dir)

    paddle.init(use_gpu=args.use_gpu, trainer_count=args.trainer_count)

    train(
        dict_size=30000,
        batch_size=args.batch_size,
        num_passes=args.num_passes,
        beam_size=args.beam_size,
        schedule_type=args.schedule_type,
        decay_a=args.decay_a,
        decay_b=args.decay_b,
        model_dir=args.model_output_dir)
