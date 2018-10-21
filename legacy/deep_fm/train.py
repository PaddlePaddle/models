import os
import gzip
import logging
import argparse

import paddle.v2 as paddle

from network_conf import DeepFM
import reader

logging.basicConfig()
logger = logging.getLogger("paddle")
logger.setLevel(logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description="PaddlePaddle DeepFM example")
    parser.add_argument(
        '--train_data_path',
        type=str,
        required=True,
        help="The path of training dataset")
    parser.add_argument(
        '--test_data_path',
        type=str,
        required=True,
        help="The path of testing dataset")
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1000,
        help="The size of mini-batch (default:1000)")
    parser.add_argument(
        '--num_passes',
        type=int,
        default=10,
        help="The number of passes to train (default: 10)")
    parser.add_argument(
        '--factor_size',
        type=int,
        default=10,
        help="The factor size for the factorization machine (default:10)")
    parser.add_argument(
        '--model_output_dir',
        type=str,
        default='models',
        help='The path for model to store (default: models)')

    return parser.parse_args()


def train():
    args = parse_args()

    if not os.path.isdir(args.model_output_dir):
        os.mkdir(args.model_output_dir)

    paddle.init(use_gpu=False, trainer_count=1)

    optimizer = paddle.optimizer.Adam(learning_rate=1e-4)

    model = DeepFM(args.factor_size)

    params = paddle.parameters.create(model)

    trainer = paddle.trainer.SGD(cost=model,
                                 parameters=params,
                                 update_equation=optimizer)

    dataset = reader.Dataset()

    def __event_handler__(event):
        if isinstance(event, paddle.event.EndIteration):
            num_samples = event.batch_id * args.batch_size
            if event.batch_id % 100 == 0:
                logger.warning("Pass %d, Batch %d, Samples %d, Cost %f, %s" %
                               (event.pass_id, event.batch_id, num_samples,
                                event.cost, event.metrics))

            if event.batch_id % 10000 == 0:
                if args.test_data_path:
                    result = trainer.test(
                        reader=paddle.batch(
                            dataset.test(args.test_data_path),
                            batch_size=args.batch_size),
                        feeding=reader.feeding)
                    logger.warning("Test %d-%d, Cost %f, %s" %
                                   (event.pass_id, event.batch_id, result.cost,
                                    result.metrics))

                path = "{}/model-pass-{}-batch-{}.tar.gz".format(
                    args.model_output_dir, event.pass_id, event.batch_id)
                with gzip.open(path, 'w') as f:
                    trainer.save_parameter_to_tar(f)

    trainer.train(
        reader=paddle.batch(
            paddle.reader.shuffle(
                dataset.train(args.train_data_path),
                buf_size=args.batch_size * 10000),
            batch_size=args.batch_size),
        feeding=reader.feeding,
        event_handler=__event_handler__,
        num_passes=args.num_passes)


if __name__ == '__main__':
    train()
