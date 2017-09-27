#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys

from paddle.v2.layer import parse_network

from paddle.trainer.config_parser import parse_config
from paddle.proto import TrainerConfig_pb2

sys.path.append('../../ctr')
import reader
import network_conf
from train import dnn_layer_dims
from utils import ModelType

dnn_input_dim, lr_input_dim = reader.load_data_meta(
    '../../ctr/output/data.meta.txt')
# create the mdoel
ctr_model = network_conf.CTRmodel(
    dnn_layer_dims,
    dnn_input_dim,
    lr_input_dim,
    model_type=ModelType(ModelType.REGRESSION),
    is_infer=True)

conf = parse_network(ctr_model.model)
sys.stdout.write(conf.SerializeToString())