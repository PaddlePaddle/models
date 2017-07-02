#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import gzip

import paddle.v2 as paddle
from network_conf import DSSM
import reader
from utils import TaskType, load_dic, logger, ModelType, ModelArch

parser = argparse.ArgumentParser(description="PaddlePaddle DSSM example")

parser.add_argument(
    '-i',
    '--train_data_path',
    type=str,
    required=False,
    help="path of training dataset")
parser.add_argument(
    '-t',
    '--test_data_path',
    type=str,
    required=False,
    help="path of testing dataset")
parser.add_argument(
    '-s',
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
    '-b',
    '--batch_size',
    type=int,
    default=10,
    help="size of mini-batch (default:10)")
parser.add_argument(
    '-p',
    '--num_passes',
    type=int,
    default=10,
    help="number of passes to run(default:10)")
parser.add_argument(
    '-y',
    '--model_type',
    type=int,
    required=True,
    default=ModelType.CLASSIFICATION_MODE,
    help="model type, %d for classification, %d for pairwise rank (default: classification)"
    % (ModelType.CLASSIFICATION_MODE, ModelType.RANK_MODE))
parser.add_argument(
    '--model_arch',
    type=int,
    required=True,
    default=ModelArch.CNN_MODE,
    help="model architecture, %d for CNN, %d for FC, %d for RNN" %
    (ModelArch.CNN_MODE, ModelArch.FC_MODE, ModelArch.RNN_MODE))
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

# arguments check.
args = parser.parse_args()
args.model_type = ModelType(args.model_type)
args.model_arch = ModelArch(args.model_arch)
if args.model_type.is_classification():
    assert args.class_num > 1, "--class_num should be set in classification task."

layer_dims = [int(i) for i in args.dnn_dims.split(',')]
target_dic_path = args.source_dic_path if not args.target_dic_path else args.target_dic_path

model_save_name_prefix = "dssm_pass_%s_%s" % (args.model_type,
                                              args.model_arch, )


def train(train_data_path=None,
          test_data_path=None,
          source_dic_path=None,
          target_dic_path=None,
          model_type=ModelType.create_classification(),
          model_arch=ModelArch.create_cnn(),
          batch_size=10,
          num_passes=10,
          share_semantic_generator=False,
          share_embed=False,
          class_num=None,
          num_workers=1,
          use_gpu=False):
    '''
    Train the DSSM.
    '''
    default_train_path = './data/rank/train.txt'
    default_test_path = './data/rank/test.txt'
    default_dic_path = './data/vocab.txt'
    if model_type.is_classification():
        default_train_path = './data/classification/train.txt'
        default_test_path = './data/classification/test.txt'

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
        model_type=model_type, )

    train_reader = paddle.batch(
        paddle.reader.shuffle(dataset.train, buf_size=1000),
        batch_size=batch_size)

    test_reader = paddle.batch(
        paddle.reader.shuffle(dataset.test, buf_size=1000),
        batch_size=batch_size)

    paddle.init(use_gpu=use_gpu, trainer_count=num_workers)

    cost, prediction, label = DSSM(
        dnn_dims=layer_dims,
        vocab_sizes=[
            len(load_dic(path)) for path in [source_dic_path, target_dic_path]
        ],
        model_type=model_type,
        model_arch=model_arch,
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
        extra_layers=paddle.evaluator.auc(input=prediction, label=label)
        if prediction else None,
        parameters=parameters,
        update_equation=adam_optimizer)

    feeding = {}
    if model_type.is_classification():
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
                logger.info("Pass %d, Batch %d, Cost %f, %s\n" % (
                    event.pass_id, event.batch_id, event.cost, event.metrics))

        if isinstance(event, paddle.event.EndPass):
            if test_reader is not None:
                if model_type.is_classification():
                    result = trainer.test(reader=test_reader, feeding=feeding)
                    logger.info("Test at Pass %d, %s \n" % (event.pass_id,
                                                            result.metrics))
                else:
                    result = None
            with gzip.open("dssm_%s_pass_%05d.tar.gz" %
                           (model_save_name_prefix, event.pass_id), "w") as f:
                parameters.to_tar(f)

    trainer.train(
        reader=train_reader,
        event_handler=_event_handler,
        feeding=feeding,
        num_passes=num_passes)

    logger.info("Training has finished.")


if __name__ == '__main__':
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
