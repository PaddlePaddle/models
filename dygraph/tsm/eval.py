#  Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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

from model import TSM_ResNet
from config_utils import *
from reader import KineticsReader

logging.root.handlers = []
FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser("Paddle Video test script")
    parser.add_argument(
        '--config',
        type=str,
        default='tsm.yaml',
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
        '--weights', type=str, default="./final", help="weight path")
    args = parser.parse_args()
    return args


def test(args):
    # parse config
    config = parse_config(args.config)
    test_config = merge_configs(config, 'test', vars(args))
    print_configs(test_config, 'Test')
    place = fluid.CUDAPlace(0)

    with fluid.dygraph.guard(place):
        video_model = TSM_ResNet("TSM", test_config)

        model_dict, _ = fluid.load_dygraph(args.weights)
        video_model.set_dict(model_dict)

        test_reader = KineticsReader(mode='test', cfg=test_config)
        test_reader = test_reader.create_reader()

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
            loss = fluid.layers.cross_entropy(
                input=outputs, label=labels, ignore_index=-1)

            avg_loss = fluid.layers.mean(loss)

            acc_top1 = fluid.layers.accuracy(input=outputs, label=labels, k=1)
            acc_top5 = fluid.layers.accuracy(input=outputs, label=labels, k=5)
            total_loss += avg_loss.numpy()
            total_acc1 += acc_top1.numpy()
            total_acc5 += acc_top5.numpy()
            total_sample += 1
            print('TEST iter {}, loss = {}, acc1 {}, acc5 {}'.format(
                batch_id, avg_loss.numpy(), acc_top1.numpy(), acc_top5.numpy()))
        print('Finish loss {}, acc1 {}, acc5 {}'.format(
            total_loss / total_sample, total_acc1 / total_sample, total_acc5 /
            total_sample))


if __name__ == "__main__":
    args = parse_args()
    logger.info(args)
    test(args)
