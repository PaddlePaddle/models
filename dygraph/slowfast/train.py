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
import paddle.fluid as fluid
from paddle.fluid.dygraph.base import to_variable
from paddle.io import DataLoader, Dataset, DistributedBatchSampler

from model import *
from config_utils import *
from lr_policy import get_epoch_lr
from kinetics_dataset import KineticsDataset
from timer import TimeAverager

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
        default=False,
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
        '--epoch',
        type=int,
        default=None,
        help='epoch number, None to read from config file')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=None,
        help='training batch size. None to use config file setting.')
    parser.add_argument(
        '--save_dir',
        type=str,
        default='./checkpoints',
        help='default model save in ./checkpoints.')
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
        '--valid_interval',
        type=int,
        default=1,
        help='validation epoch interval, 0 for no validation.')
    parser.add_argument(
        '--log_interval',
        type=int,
        default=10,
        help='mini-batch interval to log.')

    args = parser.parse_args()
    return args


def val(epoch, model, valid_loader, use_visualdl):
    val_iter_num = len(valid_loader)
    total_loss = 0.0
    total_acc1 = 0.0
    total_acc5 = 0.0
    total_sample = 0

    for batch_id, data in enumerate(valid_loader):
        y_data = data[2]
        labels = to_variable(y_data)
        labels.stop_gradient = True
        model_inputs = [data[0], data[1]]

        preds = model(model_inputs, training=False)

        loss_out = fluid.layers.softmax_with_cross_entropy(
            logits=preds, label=labels)
        avg_loss = fluid.layers.mean(loss_out)
        acc_top1 = fluid.layers.accuracy(input=preds, label=labels, k=1)
        acc_top5 = fluid.layers.accuracy(input=preds, label=labels, k=5)

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


def create_optimizer(cfg, data_size, params):
    l2_weight_decay = cfg.l2_weight_decay
    momentum = cfg.momentum

    lr_list = []
    bd_list = []
    cur_bd = 1
    for cur_epoch in range(cfg.epoch):
        for cur_iter in range(data_size):
            cur_lr = get_epoch_lr(cur_epoch + float(cur_iter) / data_size, cfg)
            lr_list.append(cur_lr)
            bd_list.append(cur_bd)
            cur_bd += 1
    bd_list.pop()

    optimizer = fluid.optimizer.Momentum(
        learning_rate=fluid.layers.piecewise_decay(
            boundaries=bd_list, values=lr_list),
        momentum=momentum,
        regularization=fluid.regularizer.L2Decay(l2_weight_decay),
        use_nesterov=True,
        parameter_list=params)

    return optimizer


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
    :param model: the model whose bn stats will be recomputed
    :param data_loader: an iterator. Produce data as input to the model
    :param num_iters: number of iterations to compute the stats.
    :return: the model with precise mean and variance in bn layers.
    """
    bn_layers_list = [
        m for m in model.sublayers()
        if isinstance(m, paddle.fluid.dygraph.nn.BatchNorm) and not m._is_test
    ]
    if len(bn_layers_list) == 0:
        return

    # moving_mean=moving_mean*momentum+batch_mean*(1.−momentum)
    # we set momentum=0. to get the true mean and variance during forward
    momentum_actual = [bn._momentum for bn in bn_layers_list]
    for bn in bn_layers_list:
        bn._momentum = 0.

    running_mean = [
        fluid.layers.zeros_like(bn._mean) for bn in bn_layers_list
    ]  #pre-ignore
    running_var = [
        fluid.layers.zeros_like(bn._variance) for bn in bn_layers_list
    ]

    ind = -1
    for ind, data in enumerate(itertools.islice(data_loader, num_iters)):
        model_inputs = [data[0], data[1]]
        model(model_inputs, training=True)

        for i, bn in enumerate(bn_layers_list):
            # Accumulates the bn stats.
            running_mean[i] += (bn._mean - running_mean[i]) / (ind + 1)
            running_var[i] += (bn._variance - running_var[i]) / (ind + 1)

    assert ind == num_iters - 1, (
        "update_bn_stats is meant to run for {} iterations, "
        "but the dataloader stops at {} iterations.".format(num_iters, ind))

    # Sets the precise bn stats.
    for i, bn in enumerate(bn_layers_list):
        bn._mean.set_value(running_mean[i])
        bn._variance.set_value(running_var[i])
        bn._momentum = momentum_actual[i]


def train(args):
    config = parse_config(args.config)
    train_config = merge_configs(config, 'train', vars(args))
    valid_config = merge_configs(config, 'valid', vars(args))
    print_configs(train_config, 'Train')

    # visual dl to visualize trianing process
    local_rank = fluid.dygraph.parallel.Env().local_rank
    if args.use_visualdl:
        try:
            from visualdl import LogWriter
            vdl_writer = LogWriter(args.vd_logdir + '/' + str(local_rank))
        except:
            print(
                "visualdl is not installed, please install visualdl if you want to use"
            )

    if not args.use_gpu:
        place = fluid.CPUPlace()
    elif not args.use_data_parallel:
        place = fluid.CUDAPlace(0)
    else:
        place = fluid.CUDAPlace(fluid.dygraph.parallel.Env().dev_id)

    random.seed(0)
    np.random.seed(0)
    paddle.framework.seed(0)
    with fluid.dygraph.guard(place):
        # 1. init net
        if args.use_data_parallel:
            strategy = fluid.dygraph.parallel.prepare_context()

        video_model = SlowFast(cfg=train_config, num_classes=400)
        if args.use_data_parallel:
            video_model = fluid.dygraph.parallel.DataParallel(
                video_model, strategy, find_unused_parameters=False)

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

        # 2. reader and optimizer
        bs_train_single = int(train_config.TRAIN.batch_size / bs_denominator)
        bs_val_single = int(train_config.VALID.batch_size / bs_denominator)
        train_data = KineticsDataset(mode="train", cfg=train_config)
        valid_data = KineticsDataset(mode="valid", cfg=valid_config)
        train_sampler = DistributedBatchSampler(
            train_data,
            batch_size=bs_train_single,
            shuffle=True,
            drop_last=True)
        train_loader = DataLoader(
            train_data,
            batch_sampler=train_sampler,
            places=place,
            feed_list=None,
            num_workers=8,
            return_list=True)
        valid_sampler = DistributedBatchSampler(
            valid_data,
            batch_size=bs_val_single,
            shuffle=False,
            drop_last=False)
        valid_loader = DataLoader(
            valid_data,
            batch_sampler=valid_sampler,
            places=place,
            feed_list=None,
            num_workers=8,
            return_list=True)

        train_iter_num = len(train_loader)
        optimizer = create_optimizer(train_config.TRAIN, train_iter_num,
                                     video_model.parameters())

        #3. load checkpoint
        if args.resume:
            saved_path = "slowfast_epoch"  #default
            model_path = saved_path + args.resume_epoch
            assert os.path.exists(model_path + ".pdparams"), \
                "Given dir {}.pdparams not exist.".format(model_path)
            assert os.path.exists(model_path + ".pdopt"), \
                "Given dir {}.pdopt not exist.".format(model_path)
            para_dict, opti_dict = fluid.dygraph.load_dygraph(model_path)
            video_model.set_dict(para_dict)
            optimizer.set_dict(opti_dict)
            if args.use_visualdl:
                # change log path otherwise log history will be overwritten
                vdl_writer = LogWriter(args.vd_logdir + args.resume_epoch + '/'
                                       + str(local_rank))

        # 4. train loop
        reader_cost_averager = TimeAverager()
        batch_cost_averager = TimeAverager()
        for epoch in range(train_config.TRAIN.epoch):
            epoch_start = time.time()
            if args.resume and epoch <= args.resume_epoch:
                print("epoch:{}<=args.resume_epoch:{}, pass".format(
                    epoch, args.resume_epoch))
                continue
            video_model.train()
            total_loss = 0.0
            total_acc1 = 0.0
            total_acc5 = 0.0
            total_sample = 0

            print('start for, Epoch {}/{} '.format(epoch,
                                                   train_config.TRAIN.epoch))
            batch_start = time.time()
            for batch_id, data in enumerate(train_loader):
                reader_cost_averager.record(time.time() - batch_start)

                y_data = data[2]
                labels = to_variable(y_data)
                labels.stop_gradient = True
                model_inputs = [data[0], data[1]]

                # 4.1.1 call net()
                preds = video_model(model_inputs, training=True)
                loss_out = fluid.layers.softmax_with_cross_entropy(
                    logits=preds, label=labels)
                avg_loss = fluid.layers.mean(loss_out)
                acc_top1 = fluid.layers.accuracy(input=preds, label=labels, k=1)
                acc_top5 = fluid.layers.accuracy(input=preds, label=labels, k=5)

                # 4.1.2 call backward()
                if args.use_data_parallel:
                    avg_loss = video_model.scale_loss(avg_loss)
                    avg_loss.backward()
                    video_model.apply_collective_grads()
                else:
                    avg_loss.backward()

                # 4.1.3 call minimize()
                optimizer.minimize(avg_loss)
                video_model.clear_gradients()

                avg_loss_value = avg_loss.numpy()[0]
                acc_top1_value = acc_top1.numpy()[0]
                acc_top5_value = acc_top5.numpy()[0]

                total_loss += avg_loss_value
                total_acc1 += acc_top1_value
                total_acc5 += acc_top5_value
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

                batch_cost_averager.record(
                    time.time() - batch_start, num_samples=bs_train_single)
                if batch_id % args.log_interval == 0:
                    print(
                        "[Epoch %d, batch %d] loss %.5f, err1 %.5f, err5 %.5f, batch_cost: %.5f sec, reader_cost: %.5f sec, ips: %.5f samples/sec"
                        %
                        (epoch, batch_id, avg_loss_value, 1.0 - acc_top1_value,
                         1. - acc_top5_value, batch_cost_averager.get_average(),
                         reader_cost_averager.get_average(),
                         batch_cost_averager.get_ips_average()))
                    reader_cost_averager.reset()
                    batch_cost_averager.reset()

                batch_start = time.time()

            train_epoch_cost = time.time() - epoch_start
            print( '[Epoch %d end] avg_loss %.5f, avg_err1 %.5f, avg_err5= %.5f, epoch_cost: %.5f sec' % \
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
            if valid_config.VALID.use_preciseBN and epoch % valid_config.VALID.preciseBN_interval == 0:
                print("do precise BN in epoch {} ...".format(epoch))
                precise_BN(video_model, train_loader,
                           min(valid_config.VALID.num_batches_preciseBN,
                               len(train_loader)))

            # 4.3 save checkpoint
            if local_rank == 0:
                if not os.path.isdir(args.save_dir):
                    os.makedirs(args.save_dir)
                model_path = os.path.join(args.save_dir,
                                          "slowfast_epoch{}".format(epoch))
                fluid.dygraph.save_dygraph(video_model.state_dict(), model_path)
                fluid.dygraph.save_dygraph(optimizer.state_dict(), model_path)
                print('save_dygraph End, Epoch {}/{} '.format(
                    epoch, train_config.TRAIN.epoch))

            # 4.4 validation
            video_model.eval()
            val(epoch, video_model, valid_loader, args.use_visualdl)

        logger.info('[TRAIN] training finished')


if __name__ == "__main__":
    args = parse_args()
    logger.info(args)
    train(args)
