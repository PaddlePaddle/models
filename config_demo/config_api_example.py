# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import multiprocessing

import numpy as np

import logging

from paddle import fluid

from ppdet.utils.stats import TrainingStats
from ppdet.utils.run_utils import parse_fetches

from ppdet.core.workspace import load_config, merge_config, create
from ppdet.data_feed import make_reader
from ppdet.utils.cli import parse_args

FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)

logger = logging.getLogger(__name__)


def main():
    args = parse_args(sys.argv[1:])
    if args.config is None:
        print("Please specify config file")
        sys.exit(1)

    config = load_config(args.config)

    if 'architecture' in config:
        main_arch = config['architecture']
    else:
        print("main architecture not given in config file")
        sys.exit(1)

    merge_config(args.cli_config)

    if config['use_gpu']:
        devices_num = fluid.core.get_cuda_device_count()
    else:
        # XXX is this intended????
        devices_num = int(
            os.environ.get('CPU_NUM', multiprocessing.cpu_count()))

    if 'train_feed' not in config:
        train_feed = create(type(main_arch).__name__ + 'TrainFeed')
    else:
        train_feed = create(config['train_feed'])

    place = fluid.CUDAPlace(0) if config['use_gpu'] else fluid.CPUPlace()
    exe = fluid.Executor(place)

    model = create(main_arch)
    lr_builder = create('LearningRate')
    optim_builder = create('OptimizerBuilder')

    startup_prog = fluid.Program()
    train_prog = fluid.Program()
    with fluid.program_guard(train_prog, startup_prog):
        with fluid.unique_name.guard():
            # XXX must be in the same scope as model when initialize feed_vars
            train_pyreader, reader, feed_vars = make_reader(
                train_feed, max_iter=config['max_iters'] * devices_num)
            train_fetches = model.train(feed_vars)
            loss = train_fetches['loss']
            lr = lr_builder()
            optimizer = optim_builder(lr)
            optimizer.minimize(loss)

    train_pyreader.decorate_sample_list_generator(reader, place)

    # parse train fetches
    train_keys, train_values = parse_fetches(train_fetches)
    train_values.append(lr)

    # 3. Compile program for multi-devices
    build_strategy = fluid.BuildStrategy()
    build_strategy.memory_optimize = False
    build_strategy.enable_inplace = False
    sync_bn = 'batch_norm_type' in config and config['batch_norm_type'] == 'SYNC_BN'
    build_strategy.sync_batch_norm = sync_bn
    train_compile_program = fluid.compiler.CompiledProgram(
        train_prog).with_data_parallel(
            loss_name=loss.name, build_strategy=build_strategy)

    exe.run(startup_prog)
    # if cfg.resume:
    #     checkpoint.load(exe, train_prog, cfg.resume)
    # elif cfg.TRAIN.PRETRAIN_WEIGHTS and cfg.MODEL.AFFINE_CHANNEL and cfg.fuse_bn:
    #     checkpoint.load_and_fusebn(exe, train_prog, cfg.TRAIN.PRETRAIN_WEIGHTS)
    # elif cfg.TRAIN.PRETRAIN_WEIGHTS:
    #     checkpoint.load(exe, train_prog, cfg.TRAIN.PRETRAIN_WEIGHTS)

    train_stats = TrainingStats(20, train_keys)
    train_pyreader.start()
    start_time = time.time()
    end_time = time.time()

    for it in range(config['max_iters']):
        start_time = end_time
        end_time = time.time()
        outs = exe.run(train_compile_program, fetch_list=train_values)
        stats = {k: np.array(v).mean() for k, v in zip(train_keys, outs[:-1])}
        train_stats.update(stats)
        logs = train_stats.log()
        strs = 'iter: {}, lr: {:.6f}, {}, time: {:.3f}'.format(
            it, np.mean(outs[-1]), logs, end_time - start_time)
        logger.info(strs)

    train_pyreader.reset()


if __name__ == '__main__':
    main()
