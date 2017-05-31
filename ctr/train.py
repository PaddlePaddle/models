#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import paddle.v2 as paddle
from paddle.v2 import layer
from paddle.v2 import data_type as dtype
from data_provider import field_index, detect_dataset, AvazuDataset

id_features_space = 100000
dnn_layer_dims = [128, 64, 32, 1]
train_data_path = './train.txt'
data_meta_info = detect_dataset(train_data_path, 500000)
batch_size = 10000
test_set_size = 10000

logging.warning('detect categorical fields in dataset %s' % train_data_path)
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

dataset = AvazuDataset(train_data_path, n_records_as_test=test_set_size)


def event_handler(event):
    if isinstance(event, paddle.event.EndIteration):
        num_samples = event.batch_id * batch_size
        if event.batch_id % 100 == 0:
            logging.warning("Pass %d, Samples %d, Cost %f" %
                            (event.pass_id, num_samples, event.cost))

        if event.batch_id % 1000 == 0:
            result = trainer.test(
                reader=paddle.batch(dataset.test, batch_size=1000),
                feeding=field_index)
            logging.warning("Test %d-%d, Cost %f" %
                            (event.pass_id, event.batch_id, result.cost))


trainer.train(
    reader=paddle.batch(
        paddle.reader.shuffle(dataset.train, buf_size=500),
        batch_size=batch_size),
    feeding=field_index,
    event_handler=event_handler,
    num_passes=100)
