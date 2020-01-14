#  Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

import os
import sys
import time
import argparse
import ast
import logging
import numpy as np
import paddle.fluid as fluid
from paddle.fluid.dygraph.base import to_variable

from utils.train_utils import train_with_pyreader
#import models.TSM.TSM_ResNet as TSM_ResNet
#from models import TSM as TSM
from tsm_res_model import TSM_ResNet 
#import TSM
from utils.config_utils import *
from reader import get_reader
from metrics import get_metrics
from utils.utility import check_cuda

logging.root.handlers = []
FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser("Paddle Video train script")
    parser.add_argument(
        '--model_name',
        type=str,
        default='AttentionCluster',
        help='name of model to train.')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/attention_cluster.txt',
        help='path to config file of model')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=None,
        help='test batch size. None to use config file setting.')
    parser.add_argument(
        '--use_gpu',
        type=ast.literal_eval,
        default=True,
        help='default use gpu.')
    parser.add_argument(
        '--weights',
        type=str,
        default="./final",
        help="weight path")
    parser.add_argument(
        '--log_interval',
        type=int,
        default=10,
        help='mini-batch interval to log.')
    args = parser.parse_args()
    return args


def test(args):
    # parse config
    config = parse_config(args.config)
    test_config = merge_configs(config, 'test', vars(args))
    print_configs(test_config, 'Test')
    #train_model = models.get_model(args.model_name, test_config, mode='test')
    place = fluid.CUDAPlace(0)

    with fluid.dygraph.guard(place):
        video_model = TSM_ResNet('TSM', layers=test_config.MODEL.num_layers,
                              class_dim=test_config.MODEL.num_classes,
                              seg_num=test_config.MODEL.seg_num)


        model_dict, _ = fluid.load_dygraph(args.weights)
        video_model.set_dict(model_dict)

        test_reader = get_reader(args.model_name.upper(), 'test', test_config) 

        video_model.eval()
        total_loss = 0.0
        total_acc1 = 0.0
        total_acc5 = 0.0
        total_sample = 0

        for batch_id, data in enumerate(test_reader()):
            x_data = np.array([item[0] for item in data])
            y_data = np.array([item[1] for item in data]).reshape([-1, 1])
                
            imgs = to_variable(x_data)
            labels = to_variable(y_data)
            labels.stop_gradient = True
            outputs = video_model(imgs)
            loss = fluid.layers.cross_entropy(input=outputs, label=labels, ignore_index=-1)

            avg_loss = fluid.layers.mean(loss)

            acc_top1 = fluid.layers.accuracy(input=outputs, label=labels, k=1)
            acc_top5 = fluid.layers.accuracy(input=outputs, label=labels, k=5)
            total_loss += avg_loss.numpy()
            total_acc1 += acc_top1.numpy()
            total_acc5 += acc_top5.numpy()
            total_sample += 1
            print('TEST iter {}, loss = {}, acc1 {}, acc5 {}'.format(batch_id, avg_loss.numpy(), acc_top1.numpy(), acc_top5.numpy()))
        print('Finish loss {}, acc1 {}, acc5 {}'.format(total_loss/ total_sample, total_acc1 / total_sample, total_acc5 / total_sample))
         

if __name__ == "__main__":
    args = parse_args()
    # check whether the installed paddle is compiled with GPU
    check_cuda(args.use_gpu)
    logger.info(args)
    test(args)
