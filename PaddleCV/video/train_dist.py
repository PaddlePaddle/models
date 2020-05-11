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
from paddle.fluid.incubate.fleet.collective import fleet, DistributedStrategy
import paddle.fluid.incubate.fleet.base.role_maker as role_maker

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
        default='configs/tsn_dist_and_dali.yaml',
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
    parser.add_argument(
        '--num_trainers',
        type=int,
        default=1,
        help='the number of trainers when used in distributed training. No need to set this, it will be set automatically'
    )
    parser.add_argument(
        '--trainer_id',
        type=int,
        default=0,
        help='trainer id when used in distributed training. No need to set this, it will be set automatically'
    )
    args = parser.parse_args()
    return args


def train(args):
    # implement distributed training by fleet
    use_fleet = True
    if use_fleet:
        role = role_maker.PaddleCloudRoleMaker(is_collective=True)
        fleet.init(role)
        args.num_trainers = int(os.getenv("PADDLE_TRAINERS_NUM", "1"))
        args.trainer_id = int(os.getenv("PADDLE_TRAINER_ID", "0"))
        print('-------------', args.num_trainers, args.trainer_id)

    if args.trainer_id == 0:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

    # parse config
    config = parse_config(args.config)
    train_config = merge_configs(config, 'train', vars(args))
    print_configs(train_config, 'Train')
    train_model = models.get_model(args.model_name, train_config, mode='train')

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

            if use_fleet:
                optimizer = fleet.distributed_optimizer(optimizer)

            optimizer.minimize(train_loss)
            train_dataloader = train_model.dataloader()

    gpu_id = int(os.environ.get('FLAGS_selected_gpus', 0))
    place = fluid.CUDAPlace(gpu_id) if args.use_gpu else fluid.CPUPlace()
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

    if use_fleet:
        compiled_train_prog = fleet.main_program
    else:
        compiled_train_prog = fluid.compiler.CompiledProgram(
            train_prog).with_data_parallel(
                loss_name=train_loss.name,
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
    train_reader = get_reader(args.model_name.upper(), 'train', train_config)

    # get metrics 
    train_metrics = get_metrics(args.model_name.upper(), 'train', train_config)

    epochs = args.epoch or train_model.epoch_num()
    exe_places = fluid.cuda_places() if args.use_gpu else fluid.cpu_places()

    train_dataloader.set_batch_generator(train_reader, places=place)

    train_with_dataloader(
        exe,
        train_prog,
        compiled_train_prog,
        train_dataloader,
        train_fetch_list,
        train_metrics,
        epochs=epochs,
        log_interval=args.log_interval,
        save_dir=args.save_dir,
        num_trainers=args.num_trainers,
        trainer_id=args.trainer_id,
        save_model_name=args.model_name,
        fix_random_seed=args.fix_random_seed,
        is_profiler=args.is_profiler,
        profiler_path=args.profiler_path)


if __name__ == "__main__":
    args = parse_args()
    # check whether the installed paddle is compiled with GPU
    check_cuda(args.use_gpu)
    check_version()
    logger.info(args)

    train(args)
