#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import logging
import paddle.v2 as paddle
from paddle.v2 import layer
from paddle.v2 import data_type as dtype
from data_provider import field_index, detect_dataset, AvazuDataset

parser = argparse.ArgumentParser(description="PaddlePaddle CTR example")
parser.add_argument(
    '--train_data_path',
    type=str,
    required=True,
    help="path of training dataset")
parser.add_argument(
    '--batch_size',
    type=int,
    default=10000,
    help="size of mini-batch (default:10000)")
parser.add_argument(
    '--test_set_size',
    type=int,
    default=10000,
    help="size of the validation dataset(default: 10000)")
parser.add_argument(
    '--num_passes', type=int, default=10, help="number of passes to train")
parser.add_argument(
    '--num_lines_to_detact',
    type=int,
    default=500000,
    help="number of records to detect dataset's meta info")

args = parser.parse_args()

dnn_layer_dims = [128, 64, 32, 1]
data_meta_info = detect_dataset(args.train_data_path, args.num_lines_to_detact)

logging.warning('detect categorical fields in dataset %s' %
                args.train_data_path)
for key, item in data_meta_info.items():
    logging.warning('    - {}\t{}'.format(key, item))

paddle.init(use_gpu=False, trainer_count=1)

# ==============================================================================
#                    input layers
# ==============================================================================
dnn_merged_input = layer.data(
    name='dnn_input',
    type=paddle.data_type.sparse_binary_vector(data_meta_info['dnn_input']))

lr_merged_input = layer.data(
    name='lr_input',
    type=paddle.data_type.sparse_binary_vector(data_meta_info['lr_input']))

click = paddle.layer.data(name='click', type=dtype.dense_vector(1))


# ==============================================================================
#                    network structure
# ==============================================================================
def build_dnn_submodel(dnn_layer_dims):
    dnn_embedding = layer.fc(input=dnn_merged_input, size=dnn_layer_dims[0])
    _input_layer = dnn_embedding
    for i, dim in enumerate(dnn_layer_dims[1:]):
        fc = layer.fc(
            input=_input_layer,
            size=dim,
            act=paddle.activation.Relu(),
            name='dnn-fc-%d' % i)
        _input_layer = fc
    return _input_layer


# config LR submodel
def build_lr_submodel():
    fc = layer.fc(
        input=lr_merged_input, size=1, name='lr', act=paddle.activation.Relu())
    return fc


# conbine DNN and LR submodels
def combine_submodels(dnn, lr):
    merge_layer = layer.concat(input=[dnn, lr])
    fc = layer.fc(
        input=merge_layer,
        size=1,
        name='output',
        # use sigmoid function to approximate ctr rate, a float value between 0 and 1.
        act=paddle.activation.Sigmoid())
    return fc


dnn = build_dnn_submodel(dnn_layer_dims)
lr = build_lr_submodel()
output = combine_submodels(dnn, lr)

# ==============================================================================
#                   cost and train period
# ==============================================================================
classification_cost = paddle.layer.multi_binary_label_cross_entropy_cost(
    input=output, label=click)

params = paddle.parameters.create(classification_cost)

optimizer = paddle.optimizer.Momentum(momentum=0.01)

trainer = paddle.trainer.SGD(
    cost=classification_cost, parameters=params, update_equation=optimizer)

dataset = AvazuDataset(
    args.train_data_path, n_records_as_test=args.test_set_size)


def event_handler(event):
    if isinstance(event, paddle.event.EndIteration):
        num_samples = event.batch_id * args.batch_size
        if event.batch_id % 100 == 0:
            logging.warning("Pass %d, Samples %d, Cost %f" %
                            (event.pass_id, num_samples, event.cost))

        if event.batch_id % 1000 == 0:
            result = trainer.test(
                reader=paddle.batch(dataset.test, batch_size=args.batch_size),
                feeding=field_index)
            logging.warning("Test %d-%d, Cost %f" %
                            (event.pass_id, event.batch_id, result.cost))


trainer.train(
    reader=paddle.batch(
        paddle.reader.shuffle(dataset.train, buf_size=500),
        batch_size=args.batch_size),
    feeding=field_index,
    event_handler=event_handler,
    num_passes=args.num_passes)
