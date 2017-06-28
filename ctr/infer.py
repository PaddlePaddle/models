#!/usr/bin/env python
# -*- coding: utf-8 -*-
import gzip
import argparse
import itertools

import paddle.v2 as paddle
import network_conf
from train import dnn_layer_dims
import reader
from utils import logger

parser = argparse.ArgumentParser(description="PaddlePaddle CTR example")
parser.add_argument(
    '--model_gz_path',
    type=str,
    required=True,
    help="path of model parameters gz file")
parser.add_argument(
    '--data_path', type=str, required=True, help="path of the dataset to infer")
parser.add_argument(
    '--prediction_output_path',
    type=str,
    required=True,
    help="path to output the prediction")
parser.add_argument(
    '--data_meta_path',
    type=str,
    default="./data.meta",
    help="path of trainset's meta info, default is ./data.meta")

args = parser.parse_args()

paddle.init(use_gpu=False, trainer_count=1)


class CTRInferer(object):
    def __init__(self, param_path):
        logger.info("create CTR model")
        self.data_meta_info, self.fields = reader.load_data_meta(
            args.data_meta_path)
        print 'fields', self.fields
        print 'data_meta_info', self.data_meta_info
        # create the mdoel
        self.ctr_model = network_conf.CTRmodel(dnn_layer_dims,
                                               self.data_meta_info['dnn_input'],
                                               self.data_meta_info['lr_input'])
        # load parameter
        logger.info("load model parameters from %s" % param_path)
        self.parameters = paddle.parameters.Parameters.from_tar(
            gzip.open(param_path, 'r'))
        self.inferer = paddle.inference.Inference(
            output_layer=self.ctr_model.model,
            parameters=self.parameters, )

    def infer(self, data_path):
        logger.info("infer data...")
        dataset = reader.AvazuDataset(
            data_path, fields=self.fields, feature_dims=self.data_meta_info)
        infer_reader = paddle.batch(dataset.infer, batch_size=1000)
        logger.warning('write predictions to %s' % args.prediction_output_path)
        output_f = open(args.prediction_output_path, 'w')
        for id, batch in enumerate(infer_reader()):
            res = self.inferer.infer(input=batch)
            predictions = [x for x in itertools.chain.from_iterable(res)]
            assert len(batch) == len(
                predictions), "predict error, %d inputs, but %d predictions" % (
                    len(batch), len(predictions))
            output_f.write('\n'.join(map(str, predictions)) + '\n')


if __name__ == '__main__':
    ctr_inferer = CTRInferer(args.model_gz_path)
    ctr_inferer.infer(args.data_path)
