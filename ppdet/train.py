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

import paddle.fluid as fluid

import logging
logger = logging.getLogger(__name__)

from args import parse_args, print_arguments
from ppdet.core.config import load_cfg, merge_cfg
from ppdet.models import Detectors
from core.optimizer import OptimizerBuilder
from ppdet.utils.stats import TrainingStats
import ppdet.utils.checkpoint


def main():
    # load config
    args = parse_args()
    print_arguments(args)
    if args.cfg_file is None:
        raise ValueError("Should specify --cfg_file=configure_file_path.")
    #print(args.cfg_file)
    cfg = load_cfg(args.cfg_file)
    merge_cfg(vars(args), cfg)
    merge_cfg({'IS_TRAIN': True}, cfg)

    # get detector and losses
    detector = Detectors.get(cfg.MODEL.TYPE)(cfg)
    fetches = detector.train()

    # get optimizer and apply minimizing
    ob = OptimizerBuilder(cfg.OPTIMIZER)
    opt = ob.get_optimizer()
    #fetches.update({'lr': ob.lr})
    loss = fetches['total_loss']
    opt.minimize(loss)

    # define executor
    place = fluid.CUDAPlace(0) if cfg.ENV.GPU else fluid.CPUPlace()

    #train_reader = reader.train(
    #    batch_size=cfg.TRAIN.im_per_batch,
    #    total_batch_size=total_batch_size,
    #    padding_total=cfg.TRAIN.padding_minibatch,
    #    shuffle=True)

    #pyreader = detector.get_pyreader()
    #pyreader.decorate_sample_list_generator(train_reader, place)

    #exe = fluid.Executor(place)
    #exe.run(fluid.default_startup_program())
    #if cfg.PRETRAIN_WEIGTS:
    #    checkpoint.load(exe, cfg.PRETRAIN_WEIGTS)

    #build_strategy= fluid.BuildStrategy()
    #build_strategy.memory_optimize = True
    #sync_bn = getattr(cfg.TRAIN, 'BATCH_NORM_TYPE', 'BN') == 'SYNC_BN'
    #build_strategy.sync_batch_norm = sync_bn
    #compile_program = fluid.compiler.CompiledProgram(
    #        fluid.default_main_program()).with_data_parallel(
    #        loss_name=tloss.name, build_strategy=build_strategy)

    #train_stats = TrainingStats(cfg.LOG_WINDOW, keys)

    #keys, values = [], []
    #for k, v in fetches:
    #    keys.append(k)
    #    values.append(v)
    #values += ob.lr

    #pyreader.start()

    #start_time = time.time()
    #end_time = time.time()
    #for it in range(cfg.max_iter):
    #    start_time = end_time
    #    end_time = time.time()
    #    outs = exe.run(compile_program, fetch_list=values)
    #    stats = {k: np.array(v).mean() for k, v in zip(keys, outs[:-1])}

    #    train_stats.update(stats)
    #    logs = train_stats.log()
    #    strs = '{}, iter: {}, lr: {:.5f}, {}, time: {:.3f}'.format(
    #        now_time(), it,
    #        np.mean(outs[-1]), logs, end_time - start_time)
    #    logger.info(strs)
    #    sys.stdout.flush()
    #    if (it + 1) % cfg.TRAIN.snapshot_iter == 0:
    #        checkpoint.save("model_iter{}".format(it))

    #pyreader.reset()


if __name__ == '__main__':
    main()
