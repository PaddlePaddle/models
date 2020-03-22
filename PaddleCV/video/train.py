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

from utils.train_utils import train_with_dataloader
import models
from utils.config_utils import *
from reader import get_reader
from metrics import get_metrics
from utils.utility import check_cuda
from utils.utility import check_version

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
        '--use_gpu',
        type=ast.literal_eval,
        default=True,
        help='default use gpu.')
    parser.add_argument(
        '--no_memory_optimize',
        action='store_true',
        default=False,
        help='whether to use memory optimize in train')
    parser.add_argument(
        '--epoch',
        type=int,
        default=None,
        help='epoch number, 0 for read from config file')
    parser.add_argument(
        '--valid_interval',
        type=int,
        default=1,
        help='validation epoch interval, 0 for no validation.')
    parser.add_argument(
        '--save_dir',
        type=str,
        default=os.path.join('data', 'checkpoints'),
        help='directory name to save train snapshoot')
    parser.add_argument(
        '--log_interval',
        type=int,
        default=10,
        help='mini-batch interval to log.')
    parser.add_argument(
        '--fix_random_seed',
        type=ast.literal_eval,
        default=False,
        help='If set True, enable continuous evaluation job.')
    # NOTE: args for profiler, used for benchmark
    parser.add_argument(
        '--profiler_path',
        type=str,
        default='./',
        help='the path to store profiler output file. used for benchmark.')
    parser.add_argument(
        '--is_profiler',
        type=int,
        default=0,
        help='the switch profiler. used for benchmark.')
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
    if args.fix_random_seed:
        startup.random_seed = 1000
        train_prog.random_seed = 1000
    with fluid.program_guard(train_prog, startup):
        with fluid.unique_name.guard():
            train_model.build_input(use_dataloader=True)
            train_model.build_model()
            # for the input, has the form [data1, data2,..., label], so train_feeds[-1] is label
            train_feeds = train_model.feeds()
            train_fetch_list = train_model.fetches()
            train_loss = train_fetch_list[0]
            optimizer = train_model.optimizer()
            optimizer.minimize(train_loss)
            train_dataloader = train_model.dataloader()

    valid_prog = fluid.Program()
    with fluid.program_guard(valid_prog, startup):
        with fluid.unique_name.guard():
            valid_model.build_input(use_dataloader=True)
            valid_model.build_model()
            valid_feeds = valid_model.feeds()
            valid_fetch_list = valid_model.fetches()
            valid_dataloader = valid_model.dataloader()

    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(startup)

    if args.resume:
        # if resume weights is given, load resume weights directly
        assert os.path.exists(args.resume + '.pdparams'), \
                "Given resume weight dir {}.pdparams not exist.".format(args.resume)
        fluid.load(train_prog, model_path=args.resume, executor=exe)
    else:
        # if not in resume mode, load pretrain weights
        if args.pretrain:
            assert os.path.exists(args.pretrain), \
                    "Given pretrain weight dir {} not exist.".format(args.pretrain)
        pretrain = args.pretrain or train_model.get_pretrain_weights()
        if pretrain:
            train_model.load_pretrain_params(exe, pretrain, train_prog, place)

    build_strategy = fluid.BuildStrategy()
    build_strategy.enable_inplace = True
    if args.model_name in ['CTCN']:
        build_strategy.enable_sequential_execution = True

    exec_strategy = fluid.ExecutionStrategy()

    compiled_train_prog = fluid.compiler.CompiledProgram(
        train_prog).with_data_parallel(
            loss_name=train_loss.name,
            build_strategy=build_strategy,
            exec_strategy=exec_strategy)
    compiled_valid_prog = fluid.compiler.CompiledProgram(
        valid_prog).with_data_parallel(
            share_vars_from=compiled_train_prog,
            build_strategy=build_strategy,
            exec_strategy=exec_strategy)

    # get reader
    bs_denominator = 1
    if args.use_gpu:
        # check number of GPUs
        gpus = os.getenv("CUDA_VISIBLE_DEVICES", "")
        if gpus == "":
            pass
        else:
            gpus = gpus.split(",")
            num_gpus = len(gpus)
            assert num_gpus == train_config.TRAIN.num_gpus, \
                   "num_gpus({}) set by CUDA_VISIBLE_DEVICES " \
                   "shoud be the same as that " \
                   "set in {}({})".format(
                   num_gpus, args.config, train_config.TRAIN.num_gpus)
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

    epochs = args.epoch or train_model.epoch_num()

    exe_places = fluid.cuda_places() if args.use_gpu else fluid.cpu_places()
    train_dataloader.set_sample_list_generator(train_reader, places=exe_places)
    valid_dataloader.set_sample_list_generator(valid_reader, places=exe_places)

    train_with_dataloader(
        exe,
        train_prog,
        compiled_train_prog,
        train_dataloader,
        train_fetch_list,
        train_metrics,
        epochs=epochs,
        log_interval=args.log_interval,
        valid_interval=args.valid_interval,
        save_dir=args.save_dir,
        save_model_name=args.model_name,
        fix_random_seed=args.fix_random_seed,
        compiled_test_prog=compiled_valid_prog,
        test_dataloader=valid_dataloader,
        test_fetch_list=valid_fetch_list,
        test_metrics=valid_metrics,
        is_profiler=args.is_profiler,
        profiler_path=args.profiler_path)


if __name__ == "__main__":
    args = parse_args()
    # check whether the installed paddle is compiled with GPU
    check_cuda(args.use_gpu)
    check_version()
    logger.info(args)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    train(args)
