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
import time
import numpy as np
import datetime
from collections import deque


def set_paddle_flags(**kwargs):
    for key, value in kwargs.items():
        if os.environ.get(key, None) is None:
            os.environ[key] = str(value)

# NOTE(paddle-dev): All of these flags should be set before
# `import paddle`. Otherwise, it would not take any effect.
set_paddle_flags(
    FLAGS_eager_delete_tensor_gb=0,  # enable GC to save memory
)

from paddle import fluid

from ppdet.core.workspace import load_config, merge_config, create
from ppdet.data.data_feed import create_reader

from ppdet.utils.eval_utils import parse_fetches, eval_run, eval_results
from ppdet.utils.stats import TrainingStats
from ppdet.utils.cli import ArgsParser
from ppdet.utils.check import check_gpu
import ppdet.utils.checkpoint as checkpoint
from ppdet.modeling.model_input import create_feed

import logging
FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)


def main():
    env = os.environ
    FLAGS.dist = 'PADDLE_TRAINER_ID' in env and 'PADDLE_TRAINERS_NUM' in env
    if FLAGS.dist:
        import random
        local_seed = (99 + int(env['PADDLE_TRAINER_ID']))
        random.seed(local_seed)
        np.random.seed(local_seed)

    cfg = load_config(FLAGS.config)
    if 'architecture' in cfg:
        main_arch = cfg.architecture
    else:
        raise ValueError("'architecture' not specified in config file.")

    merge_config(FLAGS.opt)
    if 'log_iter' not in cfg:
        cfg.log_iter = 20

    # check if set use_gpu=True in paddlepaddle cpu version
    check_gpu(cfg.use_gpu)

    if cfg.use_gpu:
        devices_num = fluid.core.get_cuda_device_count()
    else:
        devices_num = int(os.environ.get('CPU_NUM', 1))

    if 'train_feed' not in cfg:
        train_feed = create(main_arch + 'TrainFeed')
    else:
        train_feed = create(cfg.train_feed)

    if FLAGS.eval:
        if 'eval_feed' not in cfg:
            eval_feed = create(main_arch + 'EvalFeed')
        else:
            eval_feed = create(cfg.eval_feed)

    if 'FLAGS_selected_gpus' in env:
        device_id = int(env['FLAGS_selected_gpus'])
    else:
        device_id = 0
    place = fluid.CUDAPlace(device_id)
    exe = fluid.Executor(place)

    lr_builder = create('LearningRate')
    optim_builder = create('OptimizerBuilder')

    # build program
    startup_prog = fluid.Program()
    train_prog = fluid.Program()
    with fluid.program_guard(train_prog, startup_prog):
        with fluid.unique_name.guard():
            model = create(main_arch)
            train_pyreader, feed_vars = create_feed(train_feed)
            train_fetches = model.train(feed_vars)
            loss = train_fetches['loss']
            lr = lr_builder()
            optimizer = optim_builder(lr)
            optimizer.minimize(loss)

    train_reader = create_reader(train_feed, cfg.max_iters * devices_num,
                                 FLAGS.dataset_dir)
    train_pyreader.decorate_sample_list_generator(train_reader, place)

    # parse train fetches
    train_keys, train_values, _ = parse_fetches(train_fetches)
    train_values.append(lr)

    if FLAGS.eval:
        eval_prog = fluid.Program()
        with fluid.program_guard(eval_prog, startup_prog):
            with fluid.unique_name.guard():
                model = create(main_arch)
                eval_pyreader, feed_vars = create_feed(eval_feed)
                fetches = model.eval(feed_vars)
        eval_prog = eval_prog.clone(True)

        eval_reader = create_reader(eval_feed, args_path=FLAGS.dataset_dir)
        eval_pyreader.decorate_sample_list_generator(eval_reader, place)

        # parse eval fetches
        extra_keys = []
        if cfg.metric == 'COCO':
            extra_keys = ['im_info', 'im_id', 'im_shape']
        if cfg.metric == 'VOC':
            extra_keys = ['gt_box', 'gt_label', 'is_difficult']
        eval_keys, eval_values, eval_cls = parse_fetches(fetches, eval_prog,
                                                         extra_keys)

    # compile program for multi-devices
    build_strategy = fluid.BuildStrategy()
    build_strategy.enable_inplace = False
    sync_bn = getattr(model.backbone, 'norm_type', None) == 'sync_bn'
    # only enable sync_bn in multi GPU devices
    build_strategy.sync_batch_norm = sync_bn and devices_num > 1 \
        and cfg.use_gpu

    if FLAGS.dist:
        trainer_id = int(env['PADDLE_TRAINER_ID'])
        trainers = env['PADDLE_TRAINER_ENDPOINTS']
        current_endpoint = env['PADDLE_CURRENT_ENDPOINT']
        num_trainers = int(env['PADDLE_TRAINERS_NUM'])

        config = fluid.DistributeTranspilerConfig()
        config.mode = "nccl2"
        t = fluid.DistributeTranspiler(config=config)
        t.transpile(trainer_id, trainers=trainers,
                    startup_program=startup_prog,
                    current_endpoint=current_endpoint)

        exe.run(startup_prog)

        pe = fluid.ParallelExecutor(
            use_cuda=True,
            main_program=train_prog,
            loss_name=loss.name,
            num_trainers=num_trainers,
            trainer_id=trainer_id)

    else:
        exe.run(startup_prog)
        pe = fluid.ParallelExecutor(
            main_program=train_prog,
            use_cuda=True,
            loss_name=loss.name,
            build_strategy=build_strategy)

    if FLAGS.eval:
        eval_compile_program = fluid.compiler.CompiledProgram(eval_prog)

    fuse_bn = getattr(model.backbone, 'norm_type', None) == 'affine_channel'
    start_iter = 0
    if FLAGS.resume_checkpoint:
        checkpoint.load_checkpoint(exe, train_prog, FLAGS.resume_checkpoint)
        start_iter = checkpoint.global_step()
    elif cfg.pretrain_weights and fuse_bn:
        checkpoint.load_and_fusebn(exe, train_prog, cfg.pretrain_weights)
    elif cfg.pretrain_weights:
        checkpoint.load_pretrain(exe, train_prog, cfg.pretrain_weights)

    # whether output bbox is normalized in model output layer
    is_bbox_normalized = False
    if hasattr(model, 'is_bbox_normalized') and \
            callable(model.is_bbox_normalized):
        is_bbox_normalized = model.is_bbox_normalized()

    # if map_type not set, use default 11point, only use in VOC eval
    map_type = cfg.map_type if 'map_type' in cfg else '11point'

    train_stats = TrainingStats(cfg.log_smooth_window, train_keys)
    train_pyreader.start()
    start_time = time.time()
    end_time = time.time()

    cfg_name = os.path.basename(FLAGS.config).split('.')[0]
    save_dir = os.path.join(cfg.save_dir, cfg_name)
    time_stat = deque(maxlen=cfg.log_iter)
    best_box_ap_list = [0.0, 0]  #[map, iter]
    for it in range(start_iter, cfg.max_iters):
        start_time = end_time
        end_time = time.time()
        time_stat.append(end_time - start_time)
        time_cost = np.mean(time_stat)
        eta_sec = (cfg.max_iters - it) * time_cost
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        outs = exe.run(train_compile_program, fetch_list=train_values)
        stats = {k: np.array(v).mean() for k, v in zip(train_keys, outs[:-1])}
        train_stats.update(stats)
        logs = train_stats.log()
        if it % cfg.log_iter == 0:
            strs = 'iter: {}, lr: {:.6f}, {}, time: {:.3f}, eta: {}'.format(
                it, np.mean(outs[-1]), logs, time_cost, eta)
            logger.info(strs)

        if it > 0 and it % cfg.snapshot_iter == 0 or it == cfg.max_iters - 1:
            save_name = str(it) if it != cfg.max_iters - 1 else "model_final"
            checkpoint.save(exe, train_prog, os.path.join(save_dir, save_name))

            if FLAGS.eval:
                # evaluation
                results = eval_run(exe, eval_compile_program, eval_pyreader,
                                   eval_keys, eval_values, eval_cls)
                resolution = None
                if 'mask' in results[0]:
                    resolution = model.mask_head.resolution
                box_ap_stats = eval_results(
                    results, eval_feed, cfg.metric, cfg.num_classes, resolution,
                    is_bbox_normalized, FLAGS.output_eval, map_type)
                if box_ap_stats[0] > best_box_ap_list[0]:
                    best_box_ap_list[0] = box_ap_stats[0]
                    best_box_ap_list[1] = it
                    checkpoint.save(exe, train_prog,
                                    os.path.join(save_dir, "best_model"))
                logger.info("Best test box ap: {}, in iter: {}".format(
                    best_box_ap_list[0], best_box_ap_list[1]))

    train_pyreader.reset()


if __name__ == '__main__':
    parser = ArgsParser()
    parser.add_argument(
        "-r",
        "--resume_checkpoint",
        default=None,
        type=str,
        help="Checkpoint path for resuming training.")
    parser.add_argument(
        "--eval",
        action='store_true',
        default=False,
        help="Whether to perform evaluation in train")
    parser.add_argument(
        "--output_eval",
        default=None,
        type=str,
        help="Evaluation directory, default is current directory.")
    parser.add_argument(
        "-d",
        "--dataset_dir",
        default=None,
        type=str,
        help="Dataset path, same as DataFeed.dataset.dataset_dir")
    FLAGS = parser.parse_args()
    main()
