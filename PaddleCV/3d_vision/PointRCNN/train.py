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
import logging
import numpy as np
import paddle
import paddle.fluid as fluid
from paddle.fluid.layers import control_flow
import paddle.fluid.layers.learning_rate_scheduler as lr_scheduler

from models.point_rcnn import PointRCNN
from data.kitti_rcnn_reader import KittiRCNNReader
from utils.run_utils import *
from utils.config import cfg, load_config, set_config_from_list
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
        '--train_mode',
        type=str,
        default='rpn',
        required=True,
        help='specify the training mode')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        required=True,
        help='training batch size, default 16')
    parser.add_argument(
        '--epoch',
        type=int,
        default=200,
        required=True,
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
        '--data_dir',
        type=str,
        default='./data',
        help='KITTI dataset root directory')
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
        '--worker_num',
        type=int,
        default=16,
	help='multiprocess reader process num, default 16')
    parser.add_argument(
        '--log_interval',
        type=int,
        default=1,
        help='mini-batch interval to log.')
    parser.add_argument(
        '--set',
        dest='set_cfgs',
        default=None,
        nargs=argparse.REMAINDER,
        help='set extra config keys if needed.')
    args = parser.parse_args()
    return args


def train():
    args = parse_args()
    print_arguments(args)
    # check whether the installed paddle is compiled with GPU
    # PointRCNN model can only run on GPU
    check_gpu(True)

    load_config(args.cfg)
    if args.set_cfgs is not None:
        set_config_from_list(args.set_cfgs)

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
        raise NotImplementedError("unknown train mode: {}".format(args.train_mode))

    checkpoints_dir = os.path.join(args.save_dir, args.train_mode)
    if not os.path.isdir(checkpoints_dir):
        os.makedirs(checkpoints_dir)

    kitti_rcnn_reader = KittiRCNNReader(data_dir=args.data_dir,
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

    place = fluid.CUDAPlace(0)
    exe = fluid.Executor(place)

    # build model
    startup = fluid.Program()
    train_prog = fluid.Program()
    with fluid.program_guard(train_prog, startup):
        with fluid.unique_name.guard():
            train_model = PointRCNN(cfg, args.batch_size, True, 'TRAIN')
            train_model.build()
            train_loader = train_model.get_loader()
            train_feeds = train_model.get_feeds()
            train_outputs = train_model.get_outputs()
            train_loss = train_outputs['loss']
            lr = optimize(train_loss,
                          learning_rate=cfg.TRAIN.LR,
                          warmup_factor=1. / cfg.TRAIN.DIV_FACTOR,
                          decay_factor=1e-5,
                          total_step=steps_per_epoch * args.epoch,
                          warmup_pct=cfg.TRAIN.PCT_START,
                          train_prog=train_prog,
                          startup_prog=startup,
                          weight_decay=cfg.TRAIN.WEIGHT_DECAY,
                          clip_norm=cfg.TRAIN.GRAD_NORM_CLIP)
    train_keys, train_values = parse_outputs(train_outputs, 'loss')

    exe.run(startup)

    if args.resume:
        if not os.path.isdir(args.resume):
            assert os.path.exists("{}.pdparams".format(args.resume)), \
                    "Given resume weight {}.pdparams not exist.".format(args.resume)
            assert os.path.exists("{}.pdopt".format(args.resume)), \
                    "Given resume optimizer state {}.pdopt not exist.".format(args.resume)
        fluid.load(train_prog, args.resume, exe)

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
        fluid.save(prog, path)

    # get reader
    train_reader = kitti_rcnn_reader.get_multiprocess_reader(args.batch_size,
                                                             train_feeds,
                                                             proc_num=args.worker_num,
                                                             drop_last=True)
    train_loader.set_sample_list_generator(train_reader, place)

    train_stat = Stat()
    for epoch_id in range(args.resume_epoch, args.epoch):
        try:
            train_loader.start()
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
            save_model(exe, train_prog, os.path.join(checkpoints_dir, str(epoch_id)))
            train_stat.reset()
            train_periods = []
        finally:
            train_loader.reset()


if __name__ == "__main__":
    train()
