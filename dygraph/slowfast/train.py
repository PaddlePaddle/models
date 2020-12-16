#coding: utf-8
#  Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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
import itertools
import numpy as np
import random

import paddle
import paddle.distributed as dist
from paddle.io import DataLoader, DistributedBatchSampler

from model import SlowFast
from lr_policy import get_epoch_lr
from kinetics_dataset import KineticsDataset
from multigrid import MultigridSchedule
from short_sampler import DistributedShortSampler
from save_load_helper import subn_save, subn_load
from batchnorm_helper import aggregate_sub_bn_stats
from config_utils import parse_config, merge_configs, print_configs

logging.root.handlers = []
FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser("SlowFast train")
    parser.add_argument(
        '--config',
        type=str,
        default='slowfast.yaml',
        help='path to config file of model')
    parser.add_argument(
        '--use_visualdl',
        type=ast.literal_eval,
        default=True,
        help='whether to use visual dl.')
    parser.add_argument(
        '--vd_logdir',
        type=str,
        default='./vdlog',
        help='default save visualdl_log in ./vdlog.')
    parser.add_argument(
        '--use_data_parallel',
        type=ast.literal_eval,
        default=True,
        help='default use data parallel.')
    parser.add_argument(
        '--use_gpu',
        type=ast.literal_eval,
        default=True,
        help='default use gpu.')
    parser.add_argument(
        '--save_dir',
        type=str,
        default='./checkpoints',
        help='default model save in ./checkpoints.')
    parser.add_argument(
        '--save_name_prefix',
        type=str,
        default='slowfast_epoch',
        help='default saved model name prefix is slowfast_epoch ')
    parser.add_argument(
        '--resume',
        type=ast.literal_eval,
        default=False,
        help='whether to resume training')
    parser.add_argument(
        '--resume_epoch',
        type=int,
        default=100000,
        help='epoch to resume training based on latest saved checkpoints. ')
    parser.add_argument(
        '--last_mc_epoch',
        type=int,
        default=100000,
        help='epoch to resume training based on latest saved checkpoints. ')
    parser.add_argument(
        '--valid_interval',
        type=int,
        default=None,
        help='validation epoch interval, 0 for no validation. None to use config setting.'
    )
    parser.add_argument(
        '--log_interval',
        type=int,
        default=1,
        help='mini-batch interval to log.')

    args = parser.parse_args()
    return args


def precise_BN(model, data_loader, num_iters=200):
    """
    Recompute and update the batch norm stats to make them more precise. During
    training both BN stats and the weight are changing after every iteration, so
    the running average can not precisely reflect the actual stats of the
    current model.
    In this function, the BN stats are recomputed with fixed weights, to make
    the running average more precise. Specifically, it computes the true average
    of per-batch mean/variance instead of the running average.
    This is useful to improve validation accuracy.
    Args:
        model: the model whose bn stats will be recomputed
        data_loader: an iterator. Produce data as input to the model
        num_iters: number of iterations to compute the stats.
    Return:
        the model with precise mean and variance in bn layers.
    """
    bn_layers_list = [
        m for m in model.sublayers()
        if any((isinstance(m, bn_type)
                for bn_type in (paddle.nn.BatchNorm1D, paddle.nn.BatchNorm2D,
                                paddle.nn.BatchNorm3D))) and m.training
    ]
    if len(bn_layers_list) == 0:
        return

    # moving_mean=moving_mean*momentum+batch_mean*(1.−momentum)
    # we set momentum=0. to get the true mean and variance during forward
    momentum_actual = [bn._momentum for bn in bn_layers_list]
    for bn in bn_layers_list:
        bn._momentum = 0.

    running_mean = [paddle.zeros_like(bn._mean)
                    for bn in bn_layers_list]  #pre-ignore
    running_var = [paddle.zeros_like(bn._variance) for bn in bn_layers_list]

    ind = -1
    for ind, data in enumerate(itertools.islice(data_loader, num_iters)):
        print("do preciseBN {} / {}...".format(ind + 1, num_iters))
        model_inputs = [data[0], data[1]]
        model(model_inputs)

        for i, bn in enumerate(bn_layers_list):
            # Accumulates the bn stats.
            running_mean[i] += (bn._mean - running_mean[i]) / (ind + 1)
            running_var[i] += (bn._variance - running_var[i]) / (ind + 1)

    assert ind == num_iters - 1, (
        "update_bn_stats is meant to run for {} iterations, but the dataloader stops at {} iterations.".
        format(num_iters, ind))

    # Sets the precise bn stats.
    for i, bn in enumerate(bn_layers_list):
        bn._mean.set_value(running_mean[i])
        bn._variance.set_value(running_var[i])
        bn._momentum = momentum_actual[i]


def construct_loader(cfg, place):
    _nranks = dist.ParallelEnv().nranks  # num gpu
    assert _nranks == cfg.TRAIN.num_gpus, \
        "num_gpus({}) set by CUDA_VISIBLE_DEVICES" \
        "shoud be the same as that" \
        "set in {}({})".format(
            _nranks, args.config, cfg.TRAIN.num_gpus)
    bs_denominator = cfg.TRAIN.num_gpus

    bs_train_single = int(cfg.TRAIN.batch_size / bs_denominator)
    bs_val_single = int(cfg.VALID.batch_size / bs_denominator)
    train_data = KineticsDataset(mode="train", cfg=cfg)
    valid_data = KineticsDataset(mode="valid", cfg=cfg)
    if not cfg.MULTIGRID.SHORT_CYCLE:
        train_sampler = DistributedBatchSampler(
            train_data,
            batch_size=bs_train_single,
            shuffle=True,
            drop_last=True)
    else:
        # get batch size list in short cycle schedule
        bs_factor = [
            int(
                round((float(cfg.DATA.train_crop_size) / (
                    s * cfg.MULTIGRID.default_crop_size))**2))
            for s in cfg.MULTIGRID.short_cycle_factors
        ]
        single_batch_sizes = [
            bs_train_single * bs_factor[0],
            bs_train_single * bs_factor[1],
            bs_train_single,
        ]
        train_sampler = DistributedShortSampler(
            train_data,
            batch_sizes=single_batch_sizes,
            shuffle=True,
            drop_last=True)

    train_loader = DataLoader(
        train_data,
        batch_sampler=train_sampler,
        places=place,
        num_workers=cfg.TRAIN.num_workers)
    precise_bn_loader = train_loader

    valid_sampler = DistributedBatchSampler(
        valid_data, batch_size=bs_val_single, shuffle=False, drop_last=False)
    valid_loader = DataLoader(
        valid_data,
        batch_sampler=valid_sampler,
        places=place,
        num_workers=cfg.VALID.num_workers)

    return train_loader, valid_loader, precise_bn_loader


def construct_optimizer(cfg, params_list):
    l2_weight_decay = cfg.OPTIMIZER.l2_weight_decay
    momentum = cfg.OPTIMIZER.momentum

    # we set lr during each iter
    optimizer = paddle.optimizer.Momentum(
        momentum=momentum,
        weight_decay=l2_weight_decay,
        use_nesterov=True,
        parameters=params_list)

    return optimizer


def build_trainer(cfg, place):
    """
    Build training model and its associated tools, including optimizer,
    dataloaders and meters.
    Args:
        cfg (CfgNode): configs.
    Returns:
        model: training model.
        optimizer: optimizer.
        train_loader: training data loader.
        val_loader: validatoin data loader.
        precise_bn_loader: training data loader for computing
            precise BN.
    """
    # Build the video model
    video_model = SlowFast(cfg)

    train_loader, valid_loader, precise_bn_loader = construct_loader(cfg, place)
    optimizer = construct_optimizer(cfg, video_model.parameters())

    return (
        video_model,
        optimizer,
        train_loader,
        valid_loader,
        precise_bn_loader, )


def val(epoch, model, valid_loader, use_visualdl, vdl_writer=None):
    val_iter_num = len(valid_loader)
    total_loss = 0.0
    total_acc1 = 0.0
    total_acc5 = 0.0
    total_sample = 0

    for batch_id, data in enumerate(valid_loader):
        y_data = data[2]
        labels = paddle.to_tensor(y_data)
        labels.stop_gradient = True
        model_inputs = [data[0], data[1]]

        preds = model(model_inputs)

        loss_out = paddle.nn.functional.softmax_with_cross_entropy(
            logits=preds, label=labels)
        avg_loss = paddle.mean(loss_out)
        acc_top1 = paddle.metric.accuracy(input=preds, label=labels, k=1)
        acc_top5 = paddle.metric.accuracy(input=preds, label=labels, k=5)

        total_loss += avg_loss.numpy()[0]
        total_acc1 += acc_top1.numpy()[0]
        total_acc5 += acc_top5.numpy()[0]
        total_sample += 1
        if use_visualdl:
            vdl_writer.add_scalar(
                tag="val/loss",
                step=epoch * val_iter_num + batch_id,
                value=avg_loss.numpy())
            vdl_writer.add_scalar(
                tag="val/err1",
                step=epoch * val_iter_num + batch_id,
                value=1.0 - acc_top1.numpy())
            vdl_writer.add_scalar(
                tag="val/err5",
                step=epoch * val_iter_num + batch_id,
                value=1.0 - acc_top5.numpy())
        print( "[Test Epoch %d, batch %d] loss %.5f, err1 %.5f, err5 %.5f" % \
               (epoch, batch_id, avg_loss.numpy(), 1.0 - acc_top1.numpy(), 1. - acc_top5.numpy()))
    print( '[TEST Epoch %d end] avg_loss %.5f, avg_err1 %.5f, avg_err5= %.5f' % \
           (epoch, total_loss / total_sample, 1. - total_acc1 / total_sample, 1. - total_acc5 / total_sample))

    if use_visualdl:
        vdl_writer.add_scalar(
            tag="val_epoch/loss", step=epoch, value=total_loss / total_sample)
        vdl_writer.add_scalar(
            tag="val_epoch/err1",
            step=epoch,
            value=1.0 - total_acc1 / total_sample)
        vdl_writer.add_scalar(
            tag="val_epoch/err5",
            step=epoch,
            value=1.0 - total_acc5 / total_sample)


def train(args):
    config = parse_config(args.config)
    cfg = merge_configs(config, 'train', vars(args))
    print_configs(cfg, 'Train')

    # visual dl to visualize trianing process
    local_rank = dist.ParallelEnv().local_rank
    vdl_writer = None
    if args.use_visualdl:
        try:
            from visualdl import LogWriter
            vdl_writer = LogWriter(args.vd_logdir + '/' + str(local_rank))
        except:
            print(
                "visualdl is not installed, please install visualdl if you want to use"
            )

    place = 'gpu:{}'.format(dist.ParallelEnv()
                            .dev_id) if args.use_gpu else 'cpu'
    place = paddle.set_device(place)

    if args.use_data_parallel:
        dist.init_parallel_env()

    random.seed(0)
    np.random.seed(0)
    paddle.framework.seed(0)

    # Init multigrid.
    multigrid = None
    if cfg.MULTIGRID.LONG_CYCLE or cfg.MULTIGRID.SHORT_CYCLE:
        multigrid = MultigridSchedule()
        cfg = multigrid.init_multigrid(cfg)
        if cfg.MULTIGRID.LONG_CYCLE:
            if not args.resume:
                cfg, _ = multigrid.update_long_cycle(cfg, cur_epoch=0)
            else:
                cfg, _ = multigrid.update_long_cycle(
                    cfg, cur_epoch=args.last_mc_epoch)
    multi_save_epoch = [i[-1] - 1 for i in multigrid.schedule]

    #build model
    video_model = SlowFast(cfg)
    if args.use_data_parallel:
        video_model = paddle.DataParallel(video_model)

    train_loader, valid_loader, precise_bn_loader = construct_loader(cfg, place)

    optimizer = construct_optimizer(
        cfg, video_model.parameters())  #construct optimizer

    #3. load checkpoint, subn_load
    if args.resume:
        model_path = os.path.join(args.save_dir,
                                  args.save_name_prefix + str(local_rank) + '_'
                                  + "{:05d}".format(args.resume_epoch))
        subn_load(
            video_model,
            model_path,
            optimizer, )

        if args.use_visualdl:
            # change log path otherwise log history will be overwritten
            vdl_writer = LogWriter(args.vd_logdir + str(args.resume_epoch) + '/'
                                   + str(local_rank))

    # 4. train loop
    for epoch in range(cfg.OPTIMIZER.max_epoch):
        epoch_start = time.time()
        if args.resume and epoch <= args.resume_epoch:
            print("epoch:{}<=args.resume_epoch:{}, pass".format(
                epoch, args.resume_epoch))
            continue

        if cfg.MULTIGRID.LONG_CYCLE:
            cfg, changed = multigrid.update_long_cycle(cfg, epoch)
            if changed:
                print("====== Rebuild model/optimizer/loader =====")
                (
                    video_model,
                    optimizer,
                    train_loader,
                    valid_loader,
                    precise_bn_loader, ) = build_trainer(cfg, place)

                #load checkpoint after re-build model
                if epoch != 0:
                    model_path = os.path.join(
                        args.save_dir,
                        args.save_name_prefix + str(local_rank) + '_' +
                        "{:05d}".format(epoch - 1))  #checkpoint before re-build
                    subn_load(
                        video_model,
                        model_path,
                        optimizer, )

        video_model.train()
        total_loss = 0.0
        total_acc1 = 0.0
        total_acc5 = 0.0
        total_sample = 0

        train_iter_num = len(train_loader)

        print('start for, Epoch {}/{} '.format(epoch, cfg.OPTIMIZER.max_epoch))
        batch_start = time.time()
        for batch_id, data in enumerate(train_loader):
            batch_reader_end = time.time()
            current_step_lr = get_epoch_lr(
                epoch + float(batch_id) / train_iter_num, cfg)
            optimizer.set_lr(current_step_lr)
            y_data = data[2]
            labels = paddle.to_tensor(y_data)
            labels.stop_gradient = True
            model_inputs = [data[0], data[1]]

            # 4.1.1 call net()
            preds = video_model(model_inputs)
            loss_out = paddle.nn.functional.softmax_with_cross_entropy(
                logits=preds, label=labels)
            avg_loss = paddle.mean(loss_out)
            acc_top1 = paddle.metric.accuracy(input=preds, label=labels, k=1)
            acc_top5 = paddle.metric.accuracy(input=preds, label=labels, k=5)

            # 4.1.2 call backward()
            avg_loss.backward()
            optimizer.step()
            optimizer.clear_grad()

            total_loss += avg_loss.numpy()[0]
            total_acc1 += acc_top1.numpy()[0]
            total_acc5 += acc_top5.numpy()[0]
            total_sample += 1
            if args.use_visualdl:
                vdl_writer.add_scalar(
                    tag="train/loss",
                    step=epoch * train_iter_num + batch_id,
                    value=avg_loss.numpy())
                vdl_writer.add_scalar(
                    tag="train/err1",
                    step=epoch * train_iter_num + batch_id,
                    value=1.0 - acc_top1.numpy())
                vdl_writer.add_scalar(
                    tag="train/err5",
                    step=epoch * train_iter_num + batch_id,
                    value=1.0 - acc_top5.numpy())

            train_batch_cost = time.time() - batch_start
            train_reader_cost = batch_reader_end - batch_start
            batch_start = time.time()
            if batch_id % args.log_interval == 0:
                print( "[Epoch %d, batch %d] loss %.5f, err1 %.5f, err5 %.5f, lr %.5f, batch_cost: %.5f s, reader_cost: %.5f s" % \
                       (epoch, batch_id, avg_loss.numpy(), 1.0 - acc_top1.numpy(), 1. - acc_top5.numpy(), optimizer.get_lr(), train_batch_cost, train_reader_cost))

        train_epoch_cost = time.time() - epoch_start
        print( '[Epoch %d end] avg_loss %.5f, avg_err1 %.5f, avg_err5= %.5f, epoch_cost: %.5f s' % \
               (epoch, total_loss / total_sample, 1. - total_acc1 / total_sample, 1. - total_acc5 / total_sample, train_epoch_cost))
        if args.use_visualdl:
            vdl_writer.add_scalar(
                tag="train_epoch/loss",
                step=epoch,
                value=total_loss / total_sample)
            vdl_writer.add_scalar(
                tag="train_epoch/err1",
                step=epoch,
                value=1. - total_acc1 / total_sample)
            vdl_writer.add_scalar(
                tag="train_epoch/err5",
                step=epoch,
                value=1. - total_acc5 / total_sample)

        # 4.3 do preciseBN
        if cfg.VALID.use_preciseBN and (
                epoch % cfg.VALID.preciseBN_interval == 0 or
                epoch == cfg.OPTIMIZER.max_epoch - 1):
            print("do precise BN in epoch {} ...".format(epoch))
            precise_BN(video_model, precise_bn_loader,
                       min(cfg.VALID.num_batches_preciseBN,
                           len(precise_bn_loader)))

        #  aggregate sub_BN stats
        print("Aggregate sub_BatchNorm stats...")
        aggregate_sub_bn_stats(video_model)

        # 4.3 save checkpoint
        #        if local_rank == 0:
        if epoch % 10 == 0 or epoch in multi_save_epoch:
            print("Save parameters===")
            subn_save(args.save_dir,
                      args.save_name_prefix + str(local_rank) + '_', epoch,
                      video_model, optimizer)

        # 4.4 validation
        if cfg.VALID.valid_interval > 0 and (
            (epoch + 1) % cfg.VALID.valid_interval == 0 or
                epoch == cfg.OPTIMIZER.max_epoch - 1):
            video_model.eval()
            val(epoch, video_model, valid_loader, args.use_visualdl, vdl_writer)

    logger.info('[TRAIN] training finished')


if __name__ == "__main__":
    args = parse_args()
    logger.info(args)
    train(args)
#    dist.spawn(train, args=(args, ), nprocs=4)  #nprocs=1 when single card
