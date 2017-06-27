#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import gzip

import paddle.v2 as paddle
from network_conf import DSSM
import reader
from utils import TaskType, load_dic, logger

parser = argparse.ArgumentParser(description="PaddlePaddle DSSM example")

parser.add_argument(
    '--train_data_path',
    type=str,
    required=False,
    help="path of training dataset")
parser.add_argument(
    '--test_data_path',
    type=str,
    required=False,
    help="path of testing dataset")
parser.add_argument(
    '--source_dic_path',
    type=str,
    required=False,
    help="path of the source's word dic")
parser.add_argument(
    '--target_dic_path',
    type=str,
    required=False,
    help=
    "path of the target's word dic, if not set, the `source_dic_path` will be used"
)
parser.add_argument(
    '--batch_size',
    type=int,
    default=10,
    help="size of mini-batch (default:10)")
parser.add_argument(
    '--num_passes',
    type=int,
    default=10,
    help="number of passes to run(default:10)")
parser.add_argument(
    '--task_type',
    type=int,
    default=TaskType.CLASSFICATION,
    help="task type, 0 for classification, 1 for pairwise rank")
parser.add_argument(
    '--share_network_between_source_target',
    type=bool,
    default=False,
    help="whether to share network parameters between source and target")
parser.add_argument(
    '--share_embed',
    type=bool,
    default=False,
    help="whether to share word embedding between source and target")
parser.add_argument(
    '--dnn_dims',
    type=str,
    default='256,128,64,32',
    help=
    "dimentions of dnn layers, default is '256,128,64,32', which means create a 4-layer dnn, dementions of each layer is 256, 128, 64 and 32"
)
args = parser.parse_args()

default_train_path = './data/train.txt'
default_test_path = './data/test.txt'
default_dic_path = './data/dic.txt'

layer_dims = [int(i) for i in args.dnn_dims.split(',')]
target_dic_path = args.source_dic_path if not args.target_dic_path else args.target_dic_path


def train(topology,
          train_data_path=None,
          test_data_path=None,
          source_dic_path=None,
          target_dic_path=None,
          task_type=TaskType.CLASSFICATION,
          batch_size=10,
          num_passes=10,
          share_semantic_generator=False,
          share_embed=False,
          class_num=None):
    '''
    Train the DSSM.
    '''
    use_default_data = not train_data_path

    if use_default_data:
        train_data_path = default_train_path
        test_data_path = default_test_path
        source_dic_path = default_dic_path
        target_dic_path = default_dic_path

    dataset = reader.Dataset(
        train_path=train_data_path,
        test_path=test_data_path,
        source_dic_path=source_dic_path,
        target_dic_path=target_dic_path,
        task_type=task_type, )

    train_reader = paddle.batch(
        paddle.reader.shuffle(dataset.train, buf_size=1000),
        batch_size=batch_size)

    test_reader = paddle.batch(
        paddle.reader.shuffle(dataset.test, buf_size=1000),
        batch_size=batch_size)

    cost, prediction, label = DSSM(
        dnn_dims=layer_dims,
        vocab_sizes=[
            load_dic(path).size()
            for path in [train_data_path, test_data_path]
        ],
        task_type=task_type,
        share_semantic_generator=share_semantic_generator,
        class_num=class_num,
        share_embed=share_embed)()

    parameters = paddle.parameters.create(cost)

    adam_optimizer = paddle.optimizer.Adam(
        learning_rate=1e-3,
        regularization=paddle.optimizer.L2Regularization(rate=1e-3),
        model_average=paddle.optimizer.ModelAverage(average_window=0.5))

    trainer = paddle.trainer.SGD(
        cost=cost,
        extra_layers=paddle.evaluator.auc(input=prediction, label=label),
        parameters=parameters,
        update_equation=adam_optimizer)

    feeding = {}
    if task_type == TaskType.CLASSFICATION:
        feeding = {'source_input': 0, 'target_input': 1, 'label_input': 2}
    else:
        feeding = {
            'source_input': 0,
            'left_target_input': 1,
            'right_target_input': 2,
            'label_input': 3
        }

    def _event_handler(event):
        '''
        Define batch handler
        '''
        if isinstance(event, paddle.event.EndIteration):
            if event.batch_id % 100 == 0:
                logger.info("Pass %d, Batch %d, Cost %f, %s\n" %
                            (event.pass_id, event.batch_id, event.cost,
                             event.metrics))

        if isinstance(event, paddle.event.EndPass):
            if test_reader is not None:
                result = trainer.test(reader=test_reader, feeding=feeding)
                logger.info("Test at Pass %d, %s \n" % (event.pass_id,
                                                        result.metrics))
            with gzip.open("dssm_pass_%05d.tar.gz" % event.pass_id,
                           "w") as f:
                parameters.to_tar(f)

    trainer.train(
        reader=train_reader,
        event_handler=_event_handler,
        feeding=feeding,
        num_passes=num_passes)

    logger.info("Training has finished.")
