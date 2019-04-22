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
import logging
import argparse
import numpy as np
import paddle.fluid as fluid

from config import *
import models
from datareader import get_reader
from metrics import get_metrics

logging.root.handlers = []
FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
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
        '--use_gpu', type=bool, default=True, help='default use gpu.')
    parser.add_argument(
        '--weights',
        type=str,
        default=None,
        help='weight path, None to use weights from Paddle.')
    parser.add_argument(
        '--log_interval',
        type=int,
        default=1,
        help='mini-batch interval to log.')
    args = parser.parse_args()
    return args


def test(args):
    # parse config
    config = parse_config(args.config)
    test_config = merge_configs(config, 'test', vars(args))
    print_configs(test_config, "Test")

    # build model
    test_model = models.get_model(args.model_name, test_config, mode='test')
    test_model.build_input(use_pyreader=False)
    test_model.build_model()
    test_feeds = test_model.feeds()
    test_outputs = test_model.outputs()
    test_loss = test_model.loss()

    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)

    if args.weights:
        assert os.path.exists(
            args.weights), "Given weight dir {} not exist.".format(args.weights)
    weights = args.weights or test_model.get_weights()

    test_model.load_test_weights(exe, weights,
                                 fluid.default_main_program(), place)

    # get reader and metrics
    test_reader = get_reader(args.model_name.upper(), 'test', test_config)
    test_metrics = get_metrics(args.model_name.upper(), 'test', test_config)

    test_feeder = fluid.DataFeeder(place=place, feed_list=test_feeds)
    if test_loss is None:
        fetch_list = [x.name for x in test_outputs] + [test_feeds[-1].name]
    else:
        fetch_list = [test_loss.name] + [x.name for x in test_outputs
                                         ] + [test_feeds[-1].name]

    epoch_period = []
    for test_iter, data in enumerate(test_reader()):
        cur_time = time.time()
        test_outs = exe.run(fetch_list=fetch_list, feed=test_feeder.feed(data))
        period = time.time() - cur_time
        epoch_period.append(period)
        if test_loss is None:
            loss = np.zeros(1, ).astype('float32')
            pred = np.array(test_outs[0])
            label = np.array(test_outs[-1])
        else:
            loss = np.array(test_outs[0])
            pred = np.array(test_outs[1])
            label = np.array(test_outs[-1])
        test_metrics.accumulate(loss, pred, label)

        # metric here
        if args.log_interval > 0 and test_iter % args.log_interval == 0:
            info_str = '[EVAL] Batch {}'.format(test_iter)
            test_metrics.calculate_and_log_out(loss, pred, label, info_str)
    test_metrics.finalize_and_log_out("[EVAL] eval finished. ")


if __name__ == "__main__":
    args = parse_args()
    logger.info(args)

    test(args)
