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
import logging
import numpy as np
import paddle.fluid as fluid

from tools.train_utils import train_with_pyreader, train_without_pyreader
import models
from config import *
from datareader import get_reader
from metrics import get_metrics

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
        help='training batch size. None to use config file setting.')
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=None,
        help='learning rate use for training. None to use config file setting.')
    parser.add_argument(
        '--pretrain',
        type=str,
        default=None,
        help='path to pretrain weights. None to use default weights path in  ~/.paddle/weights.'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='path to resume training based on previous checkpoints. '
        'None for not resuming any checkpoints.')
    parser.add_argument(
        '--use_gpu', type=bool, default=True, help='default use gpu.')
    parser.add_argument(
        '--no_use_pyreader',
        action='store_true',
        default=False,
        help='whether to use pyreader')
    parser.add_argument(
        '--no_memory_optimize',
        action='store_true',
        default=False,
        help='whether to use memory optimize in train')
    parser.add_argument(
        '--epoch_num',
        type=int,
        default=0,
        help='epoch number, 0 for read from config file')
    parser.add_argument(
        '--valid_interval',
        type=int,
        default=1,
        help='validation epoch interval, 0 for no validation.')
    parser.add_argument(
        '--save_dir',
        type=str,
        default='checkpoints',
        help='directory name to save train snapshoot')
    parser.add_argument(
        '--log_interval',
        type=int,
        default=10,
        help='mini-batch interval to log.')
    args = parser.parse_args()
    return args


def train(args):
    # parse config
    config = parse_config(args.config)
    train_config = merge_configs(config, 'train', vars(args))
    valid_config = merge_configs(config, 'valid', vars(args))
    print_configs(train_config, 'Train')
    train_model = models.get_model(args.model_name, train_config, mode='train')
    valid_model = models.get_model(args.model_name, valid_config, mode='valid')

    # build model
    startup = fluid.Program()
    train_prog = fluid.Program()
    with fluid.program_guard(train_prog, startup):
        with fluid.unique_name.guard():
            train_model.build_input(not args.no_use_pyreader)
            train_model.build_model()
            # for the input, has the form [data1, data2,..., label], so train_feeds[-1] is label
            train_feeds = train_model.feeds()
            train_feeds[-1].persistable = True
            # for the output of classification model, has the form [pred]
            train_outputs = train_model.outputs()
            for output in train_outputs:
                output.persistable = True
            train_loss = train_model.loss()
            train_loss.persistable = True
            # outputs, loss, label should be fetched, so set persistable to be true
            optimizer = train_model.optimizer()
            optimizer.minimize(train_loss)
            train_pyreader = train_model.pyreader()

    if not args.no_memory_optimize:
        fluid.memory_optimize(train_prog)

    valid_prog = fluid.Program()
    with fluid.program_guard(valid_prog, startup):
        with fluid.unique_name.guard():
            valid_model.build_input(not args.no_use_pyreader)
            valid_model.build_model()
            valid_feeds = valid_model.feeds()
            valid_outputs = valid_model.outputs()
            valid_loss = valid_model.loss()
            valid_pyreader = valid_model.pyreader()

    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(startup)

    if args.resume:
        # if resume weights is given, load resume weights directly
        assert os.path.exists(args.resume), \
                "Given resume weight dir {} not exist.".format(args.resume)

        def if_exist(var):
            return os.path.exists(os.path.join(args.resume, var.name))

        fluid.io.load_vars(
            exe, args.resume, predicate=if_exist, main_program=train_prog)
    else:
        # if not in resume mode, load pretrain weights
        if args.pretrain:
            assert os.path.exists(args.pretrain), \
                    "Given pretrain weight dir {} not exist.".format(args.pretrain)
        pretrain = args.pretrain or train_model.get_pretrain_weights()
        if pretrain:
            train_model.load_pretrain_params(exe, pretrain, train_prog, place)

    train_exe = fluid.ParallelExecutor(
        use_cuda=args.use_gpu,
        loss_name=train_loss.name,
        main_program=train_prog)
    valid_exe = fluid.ParallelExecutor(
        use_cuda=args.use_gpu,
        share_vars_from=train_exe,
        main_program=valid_prog)

    # get reader
    bs_denominator = 1
    if (not args.no_use_pyreader) and args.use_gpu:
        bs_denominator = train_config.TRAIN.num_gpus
    train_config.TRAIN.batch_size = int(train_config.TRAIN.batch_size /
                                        bs_denominator)
    valid_config.VALID.batch_size = int(valid_config.VALID.batch_size /
                                        bs_denominator)
    train_reader = get_reader(args.model_name.upper(), 'train', train_config)
    valid_reader = get_reader(args.model_name.upper(), 'valid', valid_config)

    # get metrics 
    train_metrics = get_metrics(args.model_name.upper(), 'train', train_config)
    valid_metrics = get_metrics(args.model_name.upper(), 'valid', valid_config)

    train_fetch_list = [train_loss.name] + [x.name for x in train_outputs
                                            ] + [train_feeds[-1].name]
    valid_fetch_list = [valid_loss.name] + [x.name for x in valid_outputs
                                            ] + [valid_feeds[-1].name]

    epochs = args.epoch_num or train_model.epoch_num()

    if args.no_use_pyreader:
        train_feeder = fluid.DataFeeder(place=place, feed_list=train_feeds)
        valid_feeder = fluid.DataFeeder(place=place, feed_list=valid_feeds)
        train_without_pyreader(
            exe,
            train_prog,
            train_exe,
            train_reader,
            train_feeder,
            train_fetch_list,
            train_metrics,
            epochs=epochs,
            log_interval=args.log_interval,
            valid_interval=args.valid_interval,
            save_dir=args.save_dir,
            save_model_name=args.model_name,
            test_exe=valid_exe,
            test_reader=valid_reader,
            test_feeder=valid_feeder,
            test_fetch_list=valid_fetch_list,
            test_metrics=valid_metrics)
    else:
        train_pyreader.decorate_paddle_reader(train_reader)
        valid_pyreader.decorate_paddle_reader(valid_reader)
        train_with_pyreader(
            exe,
            train_prog,
            train_exe,
            train_pyreader,
            train_fetch_list,
            train_metrics,
            epochs=epochs,
            log_interval=args.log_interval,
            valid_interval=args.valid_interval,
            save_dir=args.save_dir,
            save_model_name=args.model_name,
            test_exe=valid_exe,
            test_pyreader=valid_pyreader,
            test_fetch_list=valid_fetch_list,
            test_metrics=valid_metrics)


if __name__ == "__main__":
    args = parse_args()
    logger.info(args)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    train(args)
