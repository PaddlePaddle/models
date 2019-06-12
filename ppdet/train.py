#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
import numpy as np

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import paddle.fluid as fluid

from ppdet.core.config import load_cfg, merge_cfg
from ppdet.models import Detectors
from ppdet.core.optimizer import OptimizerBuilder
from ppdet.utils.stats import TrainingStats, Time
from args import parse_args, print_arguments
import ppdet.utils.checkpoint as checkpoint
from ppdet.dataset.reader import Reader
import multiprocessing


def main():
    # 1. load config
    args = parse_args()
    print_arguments(args)
    if args.cfg_file is None:
        raise ValueError("Should specify --cfg_file=configure_file_path.")
    cfg = load_cfg(args.cfg_file)
    merge_cfg(vars(args), cfg)
    merge_cfg({'IS_TRAIN': True}, cfg)
    if cfg.ENV.GPU:
        devices_num = int(fluid.core.get_cuda_device_count())
    else:
        devices_num = int(
            os.environ.get('CPU_NUM', multiprocessing.cpu_count()))

    # 2. build program
    # get detector and losses
    detector = Detectors.get(cfg.MODEL.TYPE)(cfg)
    startup_prog = fluid.Program()
    train_prog = fluid.Program()
    with fluid.program_guard(train_prog, startup_prog):
        with fluid.unique_name.guard():
            fetches = detector.train()
            # get optimizer and apply minimizing
            ob = OptimizerBuilder(cfg.OPTIMIZER)
            opt = ob.get_optimizer()
            loss = fetches['loss']
            opt.minimize(loss)

    # define executor
    place = fluid.CUDAPlace(0) if cfg.ENV.GPU else fluid.CPUPlace()
    exe = fluid.Executor(place)

    # 3. Compile program for multi-devices
    keys, values = [], []
    for k, v in fetches.items():
        keys.append(k)
        v.persistable = True
        values.append(v)
    values += [ob.get_lr()]

    build_strategy = fluid.BuildStrategy()
    build_strategy.memory_optimize = False
    build_strategy.enable_inplace = False
    sync_bn = getattr(cfg.TRAIN, 'BATCH_NORM_TYPE', 'BN') == 'SYNC_BN'
    build_strategy.sync_batch_norm = sync_bn
    compile_program = fluid.compiler.CompiledProgram(
        train_prog).with_data_parallel(
            loss_name=loss.name, build_strategy=build_strategy)

    # 4. Define reader
    reader = Reader(cfg.DATA, cfg.TRANSFORM, cfg.TRAIN.MAX_ITERS * devices_num)
    train_reader = reader.train()
    pyreader = detector.get_pyreader()
    pyreader.decorate_sample_list_generator(train_reader, place)

    # 5. Load pre-trained model
    exe.run(startup_prog)
    if cfg.TRAIN.PRETRAIN_WEIGHTS:
        checkpoint.load(exe, train_prog, cfg.TRAIN.PRETRAIN_WEIGHTS)

    # 6. Run
    train_stats = TrainingStats(cfg.TRAIN.LOG_SMOOTH_WINDOW, keys)
    pyreader.start()
    start_time = time.time()
    end_time = time.time()
    save_dir = os.path.join(cfg.TRAIN.SAVE_DIR, cfg.MODEL.TYPE)
    for it in range(cfg.TRAIN.MAX_ITERS):
        start_time = end_time
        end_time = time.time()
        outs = exe.run(compile_program, fetch_list=values)
        stats = {k: np.array(v).mean() for k, v in zip(keys, outs[:-1])}
        train_stats.update(stats)
        logs = train_stats.log()
        strs = '{}, iter: {}, lr: {:.6f}, {}, time: {:.3f}'.format(
            Time(), it, np.mean(outs[-1]), logs, end_time - start_time)
        print(strs)
        sys.stdout.flush()

        # save model
        if it % cfg.TRAIN.SNAPSHOT_ITER == 0:
            checkpoint.save(exe, train_prog,
                            os.path.join(save_dir, "{}".format(it)))
    checkpoint.save(exe, train_prog, os.path.join(save_dir, "model_final"))
    pyreader.reset()


if __name__ == '__main__':
    main()
