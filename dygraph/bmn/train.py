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

import paddle
from paddle.io import DataLoader, DistributedBatchSampler
import paddle.distributed as dist
import numpy as np
import argparse
import ast
import logging
import sys
import os

from model import BMN, bmn_loss_func
from reader import BmnDataset
from config_utils import *

DATATYPE = 'float32'

logging.root.handlers = []
FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser("Paddle dynamic graph mode of BMN.")
    parser.add_argument(
        "--use_data_parallel",
        type=ast.literal_eval,
        default=False,
        help="The flag indicating whether to use data parallel mode to train the model."
    )
    parser.add_argument(
        '--config_file',
        type=str,
        default='bmn.yaml',
        help='path to config file of model')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=None,
        help='training batch size. None to use config file setting.')
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.001,
        help='learning rate use for training. None to use config file setting.')
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='filename to resume training based on previous checkpoints. '
        'None for not resuming any checkpoints.')
    parser.add_argument(
        '--use_gpu',
        type=ast.literal_eval,
        default=True,
        help='default use gpu.')
    parser.add_argument(
        '--epoch',
        type=int,
        default=9,
        help='epoch number, 0 for read from config file')
    parser.add_argument(
        '--valid_interval',
        type=int,
        default=1,
        help='validation epoch interval, 0 for no validation.')
    parser.add_argument(
        '--save_dir',
        type=str,
        default="checkpoint",
        help='path to save train snapshoot')
    parser.add_argument(
        '--log_interval',
        type=int,
        default=10,
        help='mini-batch interval to log.')
    args = parser.parse_args()
    return args


# Optimizer
def optimizer(cfg, parameter_list):
    bd = [cfg.TRAIN.lr_decay_iter]
    base_lr = cfg.TRAIN.learning_rate
    lr_decay = cfg.TRAIN.learning_rate_decay
    l2_weight_decay = cfg.TRAIN.l2_weight_decay
    lr = [base_lr, base_lr * lr_decay]
    scheduler = paddle.optimizer.lr_scheduler.PiecewiseLR(
        boundaries=bd, values=lr)
    optimizer = paddle.optimizer.Adam(
        learning_rate=scheduler,
        parameters=parameter_list,
        weight_decay=l2_weight_decay)
    return optimizer


# Validation
def val_bmn(model, val_loader, config, args):
    for batch_id, data in enumerate(val_loader):
        x_data = paddle.to_tensor(data[0])
        gt_iou_map = paddle.to_tensor(data[1])
        gt_start = paddle.to_tensor(data[2])
        gt_end = paddle.to_tensor(data[3])
        gt_iou_map.stop_gradient = True
        gt_start.stop_gradient = True
        gt_end.stop_gradient = True

        pred_bm, pred_start, pred_end = model(x_data)

        loss, tem_loss, pem_reg_loss, pem_cls_loss = bmn_loss_func(
            pred_bm, pred_start, pred_end, gt_iou_map, gt_start, gt_end, config)
        avg_loss = paddle.mean(loss)

        if args.log_interval > 0 and (batch_id % args.log_interval == 0):
            logger.info('[VALID] iter {} '.format(batch_id)
                + '\tLoss = {}, \ttem_loss = {}, \tpem_reg_loss = {}, \tpem_cls_loss = {}'.format(
                '%.04f' % avg_loss.numpy()[0], '%.04f' % tem_loss.numpy()[0], \
                '%.04f' % pem_reg_loss.numpy()[0], '%.04f' % pem_cls_loss.numpy()[0]))


# TRAIN
def train_bmn(args):
    config = parse_config(args.config_file)
    train_config = merge_configs(config, 'train', vars(args))
    valid_config = merge_configs(config, 'valid', vars(args))

    if not args.use_gpu:
        place = paddle.CPUPlace()
    elif not args.use_data_parallel:
        place = paddle.CUDAPlace(0)
    else:
        place = paddle.CUDAPlace(dist.ParallelEnv().dev_id)

    paddle.disable_static(place)
    if args.use_data_parallel:
        dist.init_parallel_env()
    bmn = BMN(train_config)
    adam = optimizer(train_config, parameter_list=bmn.parameters())

    if args.use_data_parallel:
        bmn = paddle.DataParallel(bmn)

    if args.resume:
        # if resume weights is given, load resume weights directly
        assert os.path.exists(args.resume + ".pdparams"), \
            "Given resume weight dir {} not exist.".format(args.resume)

        model, _ = paddle.load(args.resume)
        bmn.set_dict(model)

    #Reader
    bs_denominator = 1
    if args.use_gpu:
        gpus = os.getenv("CUDA_VISIBLE_DEVICES", "")
        if gpus == "":
            pass
        else:
            gpus = gpus.split(",")
            num_gpus = len(gpus)
            assert num_gpus == train_config.TRAIN.num_gpus, \
                "num_gpus({}) set by CUDA_VISIBLE_DEVICES" \
                "shoud be the same as that" \
                "set in {}({})".format(
                    num_gpus, args.config, train_config.TRAIN.num_gpus)
        bs_denominator = train_config.TRAIN.num_gpus

    bs_train_single = int(train_config.TRAIN.batch_size / bs_denominator)
    bs_val_single = int(valid_config.VALID.batch_size / bs_denominator)

    train_dataset = BmnDataset(train_config, 'train')
    val_dataset = BmnDataset(valid_config, 'valid')
    train_sampler = DistributedBatchSampler(
        train_dataset,
        batch_size=bs_train_single,
        shuffle=train_config.TRAIN.use_shuffle,
        drop_last=True)
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        places=place,
        num_workers=train_config.TRAIN.num_workers,
        return_list=True)
    val_sampler = DistributedBatchSampler(val_dataset, batch_size=bs_val_single)
    val_loader = DataLoader(
        val_dataset,
        batch_sampler=val_sampler,
        places=place,
        num_workers=valid_config.VALID.num_workers,
        return_list=True)

    for epoch in range(args.epoch):
        for batch_id, data in enumerate(train_loader):
            x_data = paddle.to_tensor(data[0])
            gt_iou_map = paddle.to_tensor(data[1])
            gt_start = paddle.to_tensor(data[2])
            gt_end = paddle.to_tensor(data[3])
            gt_iou_map.stop_gradient = True
            gt_start.stop_gradient = True
            gt_end.stop_gradient = True

            pred_bm, pred_start, pred_end = bmn(x_data)

            loss, tem_loss, pem_reg_loss, pem_cls_loss = bmn_loss_func(
                pred_bm, pred_start, pred_end, gt_iou_map, gt_start, gt_end,
                train_config)
            avg_loss = paddle.mean(loss)

            if args.use_data_parallel:
                avg_loss = bmn.scale_loss(avg_loss)
                avg_loss.backward()
                bmn.apply_collective_grads()
            else:
                avg_loss.backward()

            adam.step()
            adam.clear_grad()

            if args.log_interval > 0 and (batch_id % args.log_interval == 0):
                logger.info('[TRAIN] Epoch {}, iter {} '.format(epoch, batch_id)
                     + '\tLoss = {}, \ttem_loss = {}, \tpem_reg_loss = {}, \tpem_cls_loss = {}'.format(
                        '%.04f' % avg_loss.numpy()[0], '%.04f' % tem_loss.numpy()[0], \
                        '%.04f' % pem_reg_loss.numpy()[0], '%.04f' % pem_cls_loss.numpy()[0]))

        logger.info('[TRAIN] Epoch {} training finished'.format(epoch))

        #save
        if not os.path.isdir(args.save_dir):
            os.makedirs(args.save_dir)

        if dist.get_rank() == 0:
            save_model_name = os.path.join(
                args.save_dir, "bmn_paddle_dy" + "_epoch{}".format(epoch))
            paddle.save(bmn.state_dict(), save_model_name)

        # validation
        if args.valid_interval > 0 and (epoch + 1) % args.valid_interval == 0:
            bmn.eval()
            val_bmn(bmn, val_loader, valid_config, args)
            bmn.train()

    #save final results
    if dist.get_rank() == 0:
        save_model_name = os.path.join(args.save_dir,
                                       "bmn_paddle_dy" + "_final")
        paddle.save(bmn.state_dict(), save_model_name)
    logger.info('[TRAIN] training finished')


if __name__ == "__main__":
    args = parse_args()
    dist.spawn(train_bmn, args=(args, ), nprocs=4)
