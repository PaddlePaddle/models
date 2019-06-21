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
FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)

from paddle import fluid

from ppdet.utils.stats import TrainingStats
from ppdet.utils.cli import parse_args
from ppdet.core.workspace import load_config, merge_config, create
from ppdet.dataset.data_feed import create_reader
import ppdet.utils.checkpoint as checkpoint

from tools.placeholder import create_feeds
from tools.eval_utils import parse_fetches, eval_run, eval_results

logger = logging.getLogger(__name__)


def main():
    args = parse_args()
    cfg = load_config(args.config)

    if 'architecture' in cfg:
        main_arch = cfg['architecture']
    else:
        raise ValueError("The architecture is not specified in config file.")

    merge_config(args.cli_config)

    if cfg['use_gpu']:
        devices_num = fluid.core.get_cuda_device_count()
    else:
        devices_num = os.environ.get('CPU_NUM', multiprocessing.cpu_count())

    if 'train_feed' not in cfg:
        train_feed = create(type(main_arch).__name__ + 'TrainFeed')
    else:
        train_feed = create(cfg['train_feed'])

    if args.eval:
        if 'eval_feed' not in cfg:
            eval_feed = create(type(main_arch).__name__ + 'EvalFeed')
        else:
            eval_feed = create(cfg['eval_feed'])

    place = fluid.CUDAPlace(0) if cfg['use_gpu'] else fluid.CPUPlace()
    exe = fluid.Executor(place)

    model = create(main_arch)
    lr_builder = create('LearningRate')
    optim_builder = create('OptimizerBuilder')

    startup_prog = fluid.Program()
    train_prog = fluid.Program()
    with fluid.program_guard(train_prog, startup_prog):
        with fluid.unique_name.guard():
            train_pyreader, feed_vars = create_feeds(train_feed)
            train_fetches = model.train(feed_vars)
            loss = train_fetches['loss']
            lr = lr_builder()
            optimizer = optim_builder(lr)
            optimizer.minimize(loss)

    train_reader = create_reader(train_feed, cfg['max_iters'] * devices_num)
    train_pyreader.decorate_sample_list_generator(train_reader, place)

    # parse train fetches
    train_keys, train_values = parse_fetches(train_fetches)
    train_values.append(lr)

    if args.eval:
        eval_prog = fluid.Program()
        with fluid.program_guard(eval_prog, startup_prog):
            with fluid.unique_name.guard():
                # must be in the same scope as model when initialize feed_vars
                eval_pyreader, feed_vars = create_feeds(eval_feed)
                if cfg['metric'] == 'COCO':
                    fetches = model.test(feed_vars)
                else:
                    fetches = model.val(feed_vars)
        eval_prog = eval_prog.clone(True)

        eval_reader = create_reader(train_feed)
        eval_pyreader.decorate_sample_list_generator(eval_reader, place)

        # parse train fetches
        extra_keys = ['im_info', 'im_id'] if cfg['metric'] == 'COCO' \
                     else []
        eval_keys, eval_values = parse_fetches(fetches, eval_prog, extra_keys)

    # 3. Compile program for multi-devices
    build_strategy = fluid.BuildStrategy()
    build_strategy.memory_optimize = False
    build_strategy.enable_inplace = False
    sync_bn = getattr(model.backbone, 'norm_type') == 'sync_bn'
    build_strategy.sync_batch_norm = sync_bn
    train_compile_program = fluid.compiler.CompiledProgram(
        train_prog).with_data_parallel(
            loss_name=loss.name, build_strategy=build_strategy)
    if args.eval:
        eval_compile_program = fluid.compiler.CompiledProgram(
            eval_prog).with_data_parallel(build_strategy=build_strategy)

    exe.run(startup_prog)

    freeze_bn = getattr(model.backbone, 'freeze_norm', False)
    if args.resume_checkpoint:
        checkpoint.load_checkpoint(exe, train_prog, args.resume_checkpoint)
    elif cfg['pretrain_weights'] and freeze_bn and args.fusebn:
        checkpoint.load_and_fusebn(exe, train_prog, cfg['pretrain_weights'])
    elif cfg['pretrain_weights']:
        checkpoint.load_pretrain(exe, train_prog, cfg['pretrain_weights'])

    train_stats = TrainingStats(cfg['log_smooth_window'], train_keys)
    train_pyreader.start()
    start_time = time.time()
    end_time = time.time()

    save_dir = os.path.join(cfg['save_dir'], cfg['architecture'])
    for it in range(cfg['max_iters']):
        start_time = end_time
        end_time = time.time()
        outs = exe.run(train_compile_program, fetch_list=train_values)
        stats = {k: np.array(v).mean() for k, v in zip(train_keys, outs[:-1])}
        train_stats.update(stats)
        logs = train_stats.log()
        strs = 'iter: {}, lr: {:.6f}, {}, time: {:.3f}'.format(
            it, np.mean(outs[-1]), logs, end_time - start_time)
        logger.info(strs)

        if it > 0 and it % cfg['snapshot_iter'] == 0:
            checkpoint.save(exe, train_prog, os.path.join(save_dir, str(it)))

            if args.eval:
                # Run evaluation
                results = eval_run(exe, eval_compile_program, eval_pyreader,
                                   eval_keys, eval_values)
                # Evaluation
                eval_results(results, eval_feed, args, cfg['metric'])

    checkpoint.save(exe, train_prog, os.path.join(save_dir, "model_final"))
    train_pyreader.reset()


if __name__ == '__main__':
    main()
