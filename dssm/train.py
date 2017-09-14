#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
<<<<<<< HEAD
import gzip
=======
>>>>>>> 8b5c739a847b03ae8e7daa10f5311ef8cd12290b

import paddle.v2 as paddle
from network_conf import DSSM
import reader
<<<<<<< HEAD
from utils import TaskType, load_dic, logger
=======
from utils import TaskType, load_dic, logger, ModelType, ModelArch, display_args
>>>>>>> 8b5c739a847b03ae8e7daa10f5311ef8cd12290b

parser = argparse.ArgumentParser(description="PaddlePaddle DSSM example")

parser.add_argument(
<<<<<<< HEAD
=======
    '-i',
>>>>>>> 8b5c739a847b03ae8e7daa10f5311ef8cd12290b
    '--train_data_path',
    type=str,
    required=False,
    help="path of training dataset")
parser.add_argument(
<<<<<<< HEAD
=======
    '-t',
>>>>>>> 8b5c739a847b03ae8e7daa10f5311ef8cd12290b
    '--test_data_path',
    type=str,
    required=False,
    help="path of testing dataset")
parser.add_argument(
<<<<<<< HEAD
=======
    '-s',
>>>>>>> 8b5c739a847b03ae8e7daa10f5311ef8cd12290b
    '--source_dic_path',
    type=str,
    required=False,
    help="path of the source's word dic")
parser.add_argument(
    '--target_dic_path',
    type=str,
    required=False,
    help="path of the target's word dic, if not set, the `source_dic_path` will be used"
)
parser.add_argument(
<<<<<<< HEAD
=======
    '-b',
>>>>>>> 8b5c739a847b03ae8e7daa10f5311ef8cd12290b
    '--batch_size',
    type=int,
    default=10,
    help="size of mini-batch (default:10)")
parser.add_argument(
<<<<<<< HEAD
=======
    '-p',
>>>>>>> 8b5c739a847b03ae8e7daa10f5311ef8cd12290b
    '--num_passes',
    type=int,
    default=10,
    help="number of passes to run(default:10)")
parser.add_argument(
<<<<<<< HEAD
    '--task_type',
    type=int,
    default=TaskType.CLASSFICATION,
    help="task type, 0 for classification, 1 for pairwise rank")
=======
    '-y',
    '--model_type',
    type=int,
    required=True,
    default=ModelType.CLASSIFICATION_MODE,
    help="model type, %d for classification, %d for pairwise rank, %d for regression (default: classification)"
    % (ModelType.CLASSIFICATION_MODE, ModelType.RANK_MODE,
       ModelType.REGRESSION_MODE))
parser.add_argument(
    '-a',
    '--model_arch',
    type=int,
    required=True,
    default=ModelArch.CNN_MODE,
    help="model architecture, %d for CNN, %d for FC, %d for RNN" %
    (ModelArch.CNN_MODE, ModelArch.FC_MODE, ModelArch.RNN_MODE))
>>>>>>> 8b5c739a847b03ae8e7daa10f5311ef8cd12290b
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
<<<<<<< HEAD
    help="dimentions of dnn layers, default is '256,128,64,32', which means create a 4-layer dnn, dementions of each layer is 256, 128, 64 and 32"
)
parser.add_argument(
    '--num_workers', type=int, default=1, help="num worker threads, default 1")

args = parser.parse_args()

default_train_path = './data/rank/train.txt'
default_test_path = './data/rank/test.txt'
default_dic_path = './data/vocab.txt'
if args.task_type == TaskType.CLASSFICATION:
    default_train_path = './data/classification/train.txt'
    default_test_path = './data/classification/test.txt'

layer_dims = [int(i) for i in args.dnn_dims.split(',')]
target_dic_path = args.source_dic_path if not args.target_dic_path else args.target_dic_path
=======
    help="dimentions of dnn layers, default is '256,128,64,32', which means create a 4-layer dnn, demention of each layer is 256, 128, 64 and 32"
)
parser.add_argument(
    '--num_workers', type=int, default=1, help="num worker threads, default 1")
parser.add_argument(
    '--use_gpu',
    type=bool,
    default=False,
    help="whether to use GPU devices (default: False)")
parser.add_argument(
    '-c',
    '--class_num',
    type=int,
    default=0,
    help="number of categories for classification task.")
parser.add_argument(
    '--model_output_prefix',
    type=str,
    default="./",
    help="prefix of the path for model to store, (default: ./)")
parser.add_argument(
    '-g',
    '--num_batches_to_log',
    type=int,
    default=100,
    help="number of batches to output train log, (default: 100)")
parser.add_argument(
    '-e',
    '--num_batches_to_test',
    type=int,
    default=200,
    help="number of batches to test, (default: 200)")
parser.add_argument(
    '-z',
    '--num_batches_to_save_model',
    type=int,
    default=400,
    help="number of batches to output model, (default: 400)")

# arguments check.
args = parser.parse_args()
args.model_type = ModelType(args.model_type)
args.model_arch = ModelArch(args.model_arch)
if args.model_type.is_classification():
    assert args.class_num > 1, "--class_num should be set in classification task."

layer_dims = [int(i) for i in args.dnn_dims.split(',')]
args.target_dic_path = args.source_dic_path if not args.target_dic_path else args.target_dic_path
>>>>>>> 8b5c739a847b03ae8e7daa10f5311ef8cd12290b


def train(train_data_path=None,
          test_data_path=None,
          source_dic_path=None,
          target_dic_path=None,
<<<<<<< HEAD
          task_type=TaskType.CLASSFICATION,
=======
          model_type=ModelType.create_classification(),
          model_arch=ModelArch.create_cnn(),
>>>>>>> 8b5c739a847b03ae8e7daa10f5311ef8cd12290b
          batch_size=10,
          num_passes=10,
          share_semantic_generator=False,
          share_embed=False,
          class_num=None,
<<<<<<< HEAD
          num_workers=1):
    '''
    Train the DSSM.
    '''
=======
          num_workers=1,
          use_gpu=False):
    '''
    Train the DSSM.
    '''
    default_train_path = './data/rank/train.txt'
    default_test_path = './data/rank/test.txt'
    default_dic_path = './data/vocab.txt'
    if not model_type.is_rank():
        default_train_path = './data/classification/train.txt'
        default_test_path = './data/classification/test.txt'

>>>>>>> 8b5c739a847b03ae8e7daa10f5311ef8cd12290b
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
<<<<<<< HEAD
        task_type=task_type, )
=======
        model_type=model_type, )
>>>>>>> 8b5c739a847b03ae8e7daa10f5311ef8cd12290b

    train_reader = paddle.batch(
        paddle.reader.shuffle(dataset.train, buf_size=1000),
        batch_size=batch_size)

    test_reader = paddle.batch(
        paddle.reader.shuffle(dataset.test, buf_size=1000),
        batch_size=batch_size)

<<<<<<< HEAD
    paddle.init(use_gpu=False, trainer_count=num_workers)
=======
    paddle.init(use_gpu=use_gpu, trainer_count=num_workers)
>>>>>>> 8b5c739a847b03ae8e7daa10f5311ef8cd12290b

    cost, prediction, label = DSSM(
        dnn_dims=layer_dims,
        vocab_sizes=[
            len(load_dic(path)) for path in [source_dic_path, target_dic_path]
        ],
<<<<<<< HEAD
        task_type=task_type,
=======
        model_type=model_type,
        model_arch=model_arch,
>>>>>>> 8b5c739a847b03ae8e7daa10f5311ef8cd12290b
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
<<<<<<< HEAD
        extra_layers=paddle.evaluator.auc(input=prediction, label=label),
=======
        extra_layers=paddle.evaluator.auc(input=prediction, label=label)
        if not model_type.is_rank() else None,
>>>>>>> 8b5c739a847b03ae8e7daa10f5311ef8cd12290b
        parameters=parameters,
        update_equation=adam_optimizer)

    feeding = {}
<<<<<<< HEAD
    if task_type == TaskType.CLASSFICATION:
=======
    if model_type.is_classification() or model_type.is_regression():
>>>>>>> 8b5c739a847b03ae8e7daa10f5311ef8cd12290b
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
<<<<<<< HEAD
            if event.batch_id % 100 == 0:
                logger.info("Pass %d, Batch %d, Cost %f, %s\n" % (
                    event.pass_id, event.batch_id, event.cost, event.metrics))

        if isinstance(event, paddle.event.EndPass):
            if test_reader is not None:
                result = trainer.test(reader=test_reader, feeding=feeding)
                logger.info("Test at Pass %d, %s \n" % (event.pass_id,
                                                        result.metrics))
            with gzip.open("dssm_pass_%05d.tar.gz" % event.pass_id, "w") as f:
                parameters.to_tar(f)
=======
            # output train log
            if event.batch_id % args.num_batches_to_log == 0:
                logger.info("Pass %d, Batch %d, Cost %f, %s" % (
                    event.pass_id, event.batch_id, event.cost, event.metrics))

            # test model
            if event.batch_id > 0 and event.batch_id % args.num_batches_to_test == 0:
                if test_reader is not None:
                    if model_type.is_classification():
                        result = trainer.test(
                            reader=test_reader, feeding=feeding)
                        logger.info("Test at Pass %d, %s" % (event.pass_id,
                                                             result.metrics))
                    else:
                        result = None
            # save model
            if event.batch_id > 0 and event.batch_id % args.num_batches_to_save_model == 0:
                model_desc = "{type}_{arch}".format(
                    type=str(args.model_type), arch=str(args.model_arch))
                with open("%sdssm_%s_pass_%05d.tar" %
                          (args.model_output_prefix, model_desc,
                           event.pass_id), "w") as f:
                    parameters.to_tar(f)
>>>>>>> 8b5c739a847b03ae8e7daa10f5311ef8cd12290b

    trainer.train(
        reader=train_reader,
        event_handler=_event_handler,
        feeding=feeding,
        num_passes=num_passes)

    logger.info("Training has finished.")


if __name__ == '__main__':
<<<<<<< HEAD
    train(class_num=3)
=======
    display_args(args)
    train(
        train_data_path=args.train_data_path,
        test_data_path=args.test_data_path,
        source_dic_path=args.source_dic_path,
        target_dic_path=args.target_dic_path,
        model_type=ModelType(args.model_type),
        model_arch=ModelArch(args.model_arch),
        batch_size=args.batch_size,
        num_passes=args.num_passes,
        share_semantic_generator=args.share_network_between_source_target,
        share_embed=args.share_embed,
        class_num=args.class_num,
        num_workers=args.num_workers,
        use_gpu=args.use_gpu)
>>>>>>> 8b5c739a847b03ae8e7daa10f5311ef8cd12290b
