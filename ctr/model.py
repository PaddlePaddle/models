#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import logging
import gzip
import paddle.v2 as paddle
from paddle.v2 import layer
from paddle.v2 import data_type as dtype
from data_provider import field_index, detect_dataset, AvazuDataset


class CTRmodel(object):
    def __init__(self, dnn_layer_dims, dnn_input_dim, lr_input_dim):
        self.dnn_layer_dims = dnn_layer_dims
        self.dnn_input_dim = dnn_input_dim
        self.lr_input_dim = lr_input_dim

        self._declare_input_layers()
        self.dnn = self._build_dnn_submodel_(self.dnn_layer_dims)
        self.lr = self._build_lr_submodel_()
        self.model = self._combine_submodels_(self.dnn, self.lr)
        self.train_cost = paddle.layer.multi_binary_label_cross_entropy_cost(
            input=self.model, label=self.click)

    def _declare_input_layers(self):
        self.dnn_merged_input = layer.data(
            name='dnn_input',
            type=paddle.data_type.sparse_binary_vector(self.dnn_input_dim))

        self.lr_merged_input = layer.data(
            name='lr_input',
            type=paddle.data_type.sparse_binary_vector(self.lr_input_dim))

        self.click = paddle.layer.data(
            name='click', type=dtype.dense_vector(1))

    def _build_dnn_submodel_(self, dnn_layer_dims):
        '''
        build DNN submodel.
        '''
        dnn_embedding = layer.fc(
            input=self.dnn_merged_input, size=dnn_layer_dims[0])
        _input_layer = dnn_embedding
        for i, dim in enumerate(dnn_layer_dims[1:]):
            fc = layer.fc(
                input=_input_layer,
                size=dim,
                act=paddle.activation.Relu(),
                name='dnn-fc-%d' % i)
            _input_layer = fc
        return _input_layer

    def _build_lr_submodel_(self):
        '''
        config LR submodel
        '''
        fc = layer.fc(
            input=self.lr_merged_input,
            size=1,
            name='lr',
            act=paddle.activation.Relu())
        return fc

    def _combine_submodels_(self, dnn, lr):
        '''
        conbine DNN and LR submodels
        '''
        merge_layer = layer.concat(input=[dnn, lr])
        fc = layer.fc(
            input=merge_layer,
            size=1,
            name='output',
            # use sigmoid function to approximate ctr rate, a float value between 0 and 1.
            act=paddle.activation.Sigmoid())
        return fc
