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
import shutil
import argparse
import ast
import logging
import numpy as np
import paddle
import paddle.fluid as fluid
from paddle.fluid.layers import control_flow
from paddle.fluid.contrib.extend_optimizer import extend_with_decoupled_weight_decay
import paddle.fluid.layers.learning_rate_scheduler as lr_scheduler

from models.point_rcnn import PointRCNN
from data.kitti_rcnn_reader import KittiRCNNReader
from utils.run_utils import check_gpu, parse_outputs, Stat 
from utils.config import cfg, load_config
from utils.optimizer import optimize

logging.root.handlers = []
FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser("PointRCNN semantic segmentation train script")
    parser.add_argument(
        '--cfg',
        type=str,
        default='cfgs/default.yml',
        help='specify the config for training')
    parser.add_argument(
        '--use_gpu',
        type=ast.literal_eval,
        default=True,
        help='default use gpu.')
    parser.add_argument(
        '--train_mode',
        type=str,
        default='rpn',
        help='specify the training mode')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='training batch size, default 16')
    parser.add_argument(
        '--epoch',
        type=int,
        default=200,
        help='epoch number. default 200.')
    parser.add_argument(
        '--save_dir',
        type=str,
        default='checkpoints',
        help='directory name to save train snapshoot')
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='path to resume training based on previous checkpoints. '
        'None for not resuming any checkpoints.')
    parser.add_argument(
        '--resume_epoch',
        type=int,
        default=0,
        help='resume epoch id')
    parser.add_argument(
        '--gt_database',
        type=str,
        default='data/gt_database/train_gt_database_3level_Car.pkl',
        help='generated gt database for augmentation')
    parser.add_argument(
        '--rcnn_training_roi_dir',
        type=str,
        default=None,
	help='specify the saved rois for rcnn training when using rcnn_offline mode')
    parser.add_argument(
        '--rcnn_training_feature_dir',
        type=str,
        default=None,
	help='specify the saved features for rcnn training when using rcnn_offline mode')
    parser.add_argument(
        '--rcnn_eval_roi_dir',
        type=str,
        default=None,
	help='specify the saved rois for rcnn evaluation when using rcnn_offline mode')
    parser.add_argument(
        '--rcnn_eval_feature_dir',
        type=str,
        default=None,
	help='specify the saved features for rcnn evaluation when using rcnn_offline mode')
    parser.add_argument(
        '--log_interval',
        type=int,
        default=1,
        help='mini-batch interval to log.')
    args = parser.parse_args()
    return args


def cosine_warmup_decay(learning_rate, warmup_factor, decay_factor,
                        total_step, warmup_pct):
    def annealing_cos(start, end, pct):
	"Cosine anneal from `start` to `end` as pct goes from 0.0 to 1.0."
	cos_out = fluid.layers.cos(pct * np.pi) + 1.
        return cos_out * (start - end) / 2. + end

    warmup_start_lr = learning_rate * warmup_factor
    decay_end_lr = learning_rate * decay_factor
    warmup_step = total_step * warmup_pct

    global_step = lr_scheduler._decay_step_counter()

    lr = fluid.layers.create_global_var(
        shape=[1],
        value=float(learning_rate),
        dtype='float32',
        persistable=True,
        name="learning_rate")

    warmup_step_var = fluid.layers.fill_constant(
        shape=[1], dtype='float32', value=float(warmup_step), force_cpu=True)

    with control_flow.Switch() as switch:
        with switch.case(global_step < warmup_step_var):
            cur_lr = annealing_cos(warmup_start_lr, learning_rate,
                                   global_step / warmup_step_var)
            fluid.layers.assign(cur_lr, lr)
        with switch.case(global_step >= warmup_step_var):
            cur_lr = annealing_cos(learning_rate, decay_end_lr,
                                   (global_step - warmup_step_var) / (total_step - warmup_step))
            fluid.layers.assign(cur_lr, lr)

    return lr


def train():
    args = parse_args()
    # check whether the installed paddle is compiled with GPU
    check_gpu(args.use_gpu)

    load_config(args.cfg)

    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    if args.train_mode == 'rpn':
        cfg.RPN.ENABLED = True
        cfg.RCNN.ENABLED = False
    elif args.train_mode == 'rcnn':
        cfg.RCNN.ENABLED = True
        cfg.RPN.ENABLED = cfg.RPN.FIXED = True
    elif args.train_mode == 'rcnn_offline':
        cfg.RCNN.ENABLED = True
        cfg.RPN.ENABLED = False
    else:
        raise NotImplementedError

    kitti_rcnn_reader = KittiRCNNReader(data_dir='./data',
                                    npoints=cfg.RPN.NUM_POINTS,
                                    split=cfg.TRAIN.SPLIT,
                                    mode='TRAIN',
                                    classes=cfg.CLASSES,
                                    rcnn_training_roi_dir=args.rcnn_training_roi_dir,
                                    rcnn_training_feature_dir=args.rcnn_training_feature_dir,
                                    gt_database_dir=args.gt_database)
    num_samples = len(kitti_rcnn_reader)
    steps_per_epoch = int(num_samples / args.batch_size)
    logger.info("Total {} samples, {} batch per epoch.".format(num_samples, steps_per_epoch))
    boundaries = [i * steps_per_epoch for i in cfg.TRAIN.DECAY_STEP_LIST]
    values = [cfg.TRAIN.LR * (cfg.TRAIN.LR_DECAY ** i) for i in range(len(boundaries) + 1)]

    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)

    # build model
    startup = fluid.Program()
    train_prog = fluid.Program()
    with fluid.program_guard(train_prog, startup):
        with fluid.unique_name.guard():
            train_model = PointRCNN(cfg, args.batch_size, True, 'TRAIN')
            train_model.build()
            train_pyreader = train_model.get_pyreader()
            train_feeds = train_model.get_feeds()
            train_outputs = train_model.get_outputs()
            train_loss = train_outputs['loss']
            # # fluid.clip.set_gradient_clip(fluid.clip.GradientClipByGlobalNorm(clip_norm=cfg.TRAIN.GRAD_NORM_CLIP))
            # # AdamW = extend_with_decoupled_weight_decay(fluid.optimizer.Adam)
            # # optimizer = AdamW(
            # optimizer = fluid.optimizer.Adam(
            #     # learning_rate=cosine_warmup_decay(
            #     #     learning_rate=cfg.TRAIN.LR,
            #     #     warmup_factor=1. / cfg.TRAIN.DIV_FACTOR,
            #     #     decay_factor=1e-5,
            #     #     total_step=steps_per_epoch * args.epoch,
            #     #     warmup_pct=cfg.TRAIN.PCT_START),
            #     learning_rate=fluid.layers.linear_lr_warmup(
            #         learning_rate=fluid.layers.piecewise_decay(
            #             boundaries=boundaries, values=values),
            #             warmup_steps=cfg.TRAIN.WARMUP_EPOCH * steps_per_epoch,
            #             start_lr=cfg.TRAIN.WARMUP_MIN,
            #             end_lr=cfg.TRAIN.LR),
            #     # learning_rate=fluid.layers.linear_lr_warmup(
            #     #     learning_rate=exponential_with_clip(
            #     #         cfg.TRAIN.LR,
            #     #         [0] + cfg.TRAIN.DECAY_STEP_LIST,
            #     #         cfg.TRAIN.LR_DECAY,
            #     #         cfg.TRAIN.LR_CLIP),
            #     #     warmup_steps=cfg.TRAIN.WARMUP_EPOCH,
            #     #     start_lr=cfg.TRAIN.WARMUP_MIN,
            #     #     end_lr=cfg.TRAIN.LR),
            #     beta1=0.9, beta2=0.99,
            #     regularization=fluid.regularizer.L2Decay(cfg.TRAIN.WEIGHT_DECAY))
            #     # weight_decay=cfg.TRAIN.WEIGHT_DECAY)
            # optimizer.minimize(train_loss)
            # lr = optimizer._global_learning_rate()
            # logger.info("Optimizer learning rate name: {}".format(lr.name))
            # print("optimizer betas:", optimizer._beta1, optimizer._beta2)
            lr = optimize(train_loss,
                                    learning_rate=cfg.TRAIN.LR,
                                    warmup_factor=1. / cfg.TRAIN.DIV_FACTOR,
                                    decay_factor=1e-5,
                                    total_step=steps_per_epoch * args.epoch,
                                    warmup_pct=cfg.TRAIN.PCT_START,
                                    train_program=train_prog,
                                    startup_prog=startup,
                                    weight_decay=cfg.TRAIN.WEIGHT_DECAY,
                                    clip_norm=cfg.TRAIN.GRAD_NORM_CLIP)
    # prog_clip = train_prog.clone()
    # train_loss_clip = prog_clip.block(0).var(train_loss.name)
    # p_g_clip = fluid.backward.append_backward(loss=train_loss_clip)
    # with fluid.program_guard(main_program=prog_clip):
	# fluid.clip.set_gradient_clip(
	#     fluid.clip.GradientClipByGlobalNorm(clip_norm=cfg.TRAIN.GRAD_NORM_CLIP))
	# p_g_clip = fluid.clip.append_gradient_clip_ops(p_g_clip)
    #
    train_keys, train_values = parse_outputs(train_outputs, 'loss')

    exe.run(startup)

    if args.resume:
        assert os.path.exists(args.resume), \
                "Given resume weight dir {} not exist.".format(args.resume)
        def if_exist(var):
            # logger.info("{}: {}".format(var.name, os.path.exists(os.path.join(args.resume, var.name))))
            return os.path.exists(os.path.join(args.resume, var.name))
        fluid.io.load_vars(
            exe, args.resume, predicate=if_exist, main_program=train_prog)

    build_strategy = fluid.BuildStrategy()
    build_strategy.memory_optimize = False
    build_strategy.enable_inplace = False
    build_strategy.fuse_all_optimizer_ops = False
    train_compile_prog = fluid.compiler.CompiledProgram(
            train_prog).with_data_parallel(loss_name=train_loss.name,
                    build_strategy=build_strategy)

    def save_model(exe, prog, path):
        if os.path.isdir(path):
            shutil.rmtree(path)
        logger.info("Save model to {}".format(path))
        fluid.io.save_persistables(exe, path, prog)

    # get reader
    train_reader = kitti_rcnn_reader.get_reader(args.batch_size, train_feeds, True)
    train_pyreader.decorate_sample_list_generator(train_reader, place)

    train_stat = Stat()
    for epoch_id in range(args.resume_epoch, args.epoch):
        try:
            train_pyreader.start()
            train_iter = 0
            train_periods = []
            while True:
                cur_time = time.time()
                train_outs = exe.run(train_compile_prog, fetch_list=train_values + [lr.name])
                period = time.time() - cur_time
                train_periods.append(period)
                train_stat.update(train_keys, train_outs[:-1])
                if train_iter % args.log_interval == 0:
                    log_str = ""
                    for name, values in zip(train_keys + ['learning_rate'], train_outs):
                        log_str += "{}: {:.6f}, ".format(name, np.mean(values))
                    logger.info("[TRAIN] Epoch {}, batch {}: {}time: {:.2f}".format(epoch_id, train_iter, log_str, period))
                train_iter += 1
        except fluid.core.EOFException:
            logger.info("[TRAIN] Epoch {} finished, {}average time: {:.2f}".format(epoch_id, train_stat.get_mean_log(), np.mean(train_periods[2:])))
            save_model(exe, train_prog, os.path.join(args.save_dir, str(epoch_id)))
            train_stat.reset()
            train_periods = []
        finally:
            train_pyreader.reset()


if __name__ == "__main__":
    train()
