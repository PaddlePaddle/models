#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import gzip
import paddle.v2 as paddle
import argparse
import cPickle

from reader import Reader
from network_conf import DNNmodel
from utils import logger


def parse_args():
    """
    parse arguments
    """
    parser = argparse.ArgumentParser(
        description="PaddlePaddle Youtube Recall Model Example")
    parser.add_argument(
        '--train_set_path',
        type=str,
        required=True,
        help="path of the train set")
    parser.add_argument(
        '--test_set_path', type=str, required=True, help="path of the test set")
    parser.add_argument(
        '--model_output_dir',
        type=str,
        required=True,
        help="directory to output")
    parser.add_argument(
        '--feature_dict',
        type=str,
        required=True,
        help="path of feature_dict.pkl")
    parser.add_argument(
        '--item_freq', type=str, required=True, help="path of item_freq.pkl ")
    parser.add_argument(
        '--window_size', type=int, default=20, help="window size(default: 20)")
    parser.add_argument(
        '--num_passes', type=int, default=1, help="number of passes to train")
    parser.add_argument(
        '--batch_size',
        type=int,
        default=50,
        help="size of mini-batch (default:50)")
    return parser.parse_args()


def train():
    """
    train
    """
    args = parse_args()

    # check argument
    assert os.path.exists(
        args.train_set_path), 'The train_set_path path does not exist.'
    assert os.path.exists(
        args.test_set_path), 'The test_set_path path does not exist.'
    assert os.path.exists(
        args.feature_dict), 'The feature_dict path does not exist.'
    assert os.path.exists(args.item_freq), 'The item_freq path does not exist.'
    assert os.path.exists(
        args.model_output_dir), 'The model_output_dir path does not exist.'

    paddle.init(use_gpu=False, trainer_count=1)

    with open(args.feature_dict) as f:
        feature_dict = cPickle.load(f)

    with open(args.item_freq) as f:
        item_freq = cPickle.load(f)

    feeding = {
        'user_id': 0,
        'province': 1,
        'city': 2,
        'history_clicked_items': 3,
        'history_clicked_categories': 4,
        'history_clicked_tags': 5,
        'phone': 6,
        'target_item': 7
    }
    optimizer = paddle.optimizer.AdaGrad(
        learning_rate=1e-1,
        regularization=paddle.optimizer.L2Regularization(rate=1e-3))

    cost = DNNmodel(
        dnn_layer_dims=[256, 31],
        feature_dict=feature_dict,
        item_freq=item_freq,
        is_infer=False).model_cost
    parameters = paddle.parameters.create(cost)

    trainer = paddle.trainer.SGD(cost, parameters, optimizer)

    def event_handler(event):
        """
        event handler
        """
        if isinstance(event, paddle.event.EndIteration):
            if event.batch_id and not event.batch_id % 10:
                logger.info("Pass %d, Batch %d, Cost %f" %
                            (event.pass_id, event.batch_id, event.cost))
        elif isinstance(event, paddle.event.EndPass):
            save_path = os.path.join(args.model_output_dir,
                                     "model_pass_%05d.tar.gz" % event.pass_id)
            logger.info("Save model into %s ..." % save_path)
            with gzip.open(save_path, "w") as f:
                trainer.save_parameter_to_tar(f)

    reader = Reader(feature_dict, args.window_size)
    trainer.train(
        paddle.batch(
            paddle.reader.shuffle(
                lambda: reader.train(args.train_set_path), buf_size=7000),
            args.batch_size),
        num_passes=args.num_passes,
        feeding=feeding,
        event_handler=event_handler)


if __name__ == "__main__":
    train()
