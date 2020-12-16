#  Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import time
import argparse
import wget
import tarfile
import logging
import numpy as np
import glob
import ast
from model import TSN_ResNet
from utils.config_utils import *
from reader.ucf101_reader import UCF101Reader
from timer import TimeAverager

import paddle
from paddle.io import DataLoader, DistributedBatchSampler
from compose import TSN_UCF101_Dataset
import paddle.nn.functional as F

logging.root.handlers = []
FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser("Paddle Video train script")
    parser.add_argument(
        '--config',
        type=str,
        default='tsn.yaml',
        help='path to config file of model')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=None,
        help='training batch size. None to use config file setting.')
    parser.add_argument(
        '--pretrain',
        type=str,
        default=None,
        help='path to pretrain weights. None to use default weights path in  ~/.paddle/weights.'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='path to resume training based on previous checkpoints. '
        'None for not resuming any checkpoints.')
    parser.add_argument(
        '--use_gpu',
        type=ast.literal_eval,
        default=True,
        help='default use gpu.')
    parser.add_argument(
        '--epoch',
        type=int,
        default=None,
        help='epoch number, 0 for read from config file')
    parser.add_argument(
        '--use_data_parallel',
        type=ast.literal_eval,
        default=True,
        help='default use data parallel.')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default="./checkpoint",
        help='path to resume training based on previous checkpoints. '
        'None for not resuming any checkpoints.')
    parser.add_argument(
        '--weights',
        type=str,
        default="./weights",
        help='path to save the final optimized model.'
        'default path is "./weights".')
    parser.add_argument(
        '--validate',
        type=str,
        default=True,
        help='whether to validating in training phase.'
        'default value is True.')
    parser.add_argument(
        '--log_interval',
        type=int,
        default=10,
        help='mini-batch interval to log.')
    args = parser.parse_args()
    return args


def init_model(model, pre_state_dict):
    param_state_dict = {}
    model_dict = model.state_dict()
    for key in model_dict.keys():
        weight_name = model_dict[key].name
        if weight_name in pre_state_dict.keys(
        ) and weight_name != "fc_0.w_0" and weight_name != "fc_0.b_0":
            print('Load weight: {}, shape: {}'.format(
                weight_name, pre_state_dict[weight_name].shape))
            param_state_dict[key] = pre_state_dict[weight_name]
        else:
            param_state_dict[key] = model_dict[key]
    model.set_dict(param_state_dict)
    return model


def val(epoch, model, val_loader, cfg, args):
    total_loss = 0.0
    total_acc1 = 0.0
    total_acc5 = 0.0
    total_sample = 0

    for batch_id, data in enumerate(val_loader):
        imgs = paddle.to_tensor(data[0])
        labels = paddle.to_tensor(data[1])
        labels.stop_gradient = True

        outputs = model(imgs)

        loss = F.cross_entropy(input=outputs, label=labels, ignore_index=-1)
        avg_loss = paddle.mean(loss)
        acc_top1 = paddle.metric.accuracy(input=outputs, label=labels, k=1)
        acc_top5 = paddle.metric.accuracy(input=outputs, label=labels, k=5)

        dy_out = avg_loss.numpy()[0]
        total_loss += dy_out
        total_acc1 += acc_top1.numpy()[0]
        total_acc5 += acc_top5.numpy()[0]
        total_sample += 1

        if batch_id % 5 == 0:
            print(
                "TEST Epoch {}, iter {}, loss={:.5f}, acc1 {:.5f}, acc5 {:.5f}".
                format(epoch, batch_id, total_loss / total_sample, total_acc1 /
                       total_sample, total_acc5 / total_sample))

    print('Finish loss {} , acc1 {} , acc5 {}'.format(
        total_loss / total_sample, total_acc1 / total_sample, total_acc5 /
        total_sample))
    return total_acc1 / total_sample


def create_optimizer(cfg, params):
    total_videos = cfg.total_videos
    step = int(total_videos / cfg.batch_size + 1)
    bd = [e * step for e in cfg.decay_epochs]
    base_lr = cfg.learning_rate
    lr_decay = cfg.learning_rate_decay
    lr = [base_lr, base_lr * lr_decay, base_lr * lr_decay * lr_decay]
    l2_weight_decay = cfg.l2_weight_decay
    momentum = cfg.momentum

    optimizer = paddle.optimizer.Momentum(
        learning_rate=paddle.optimizer.lr.PiecewiseDecay(
            boundaries=bd, values=lr),
        momentum=momentum,
        weight_decay=paddle.regularizer.L2Decay(l2_weight_decay),
        parameters=params)

    return optimizer


def train(args):
    config = parse_config(args.config)
    train_config = merge_configs(config, 'train', vars(args))
    valid_config = merge_configs(config, 'valid', vars(args))
    print_configs(train_config, 'Train')
    use_data_parallel = args.use_data_parallel

    paddle.disable_static(paddle.CUDAPlace(0))

    place = paddle.CUDAPlace(paddle.distributed.ParallelEnv().dev_id) \
        if use_data_parallel else paddle.CUDAPlace(0)
    if use_data_parallel:
        paddle.distributed.init_parallel_env()

    video_model = TSN_ResNet(train_config)
    if use_data_parallel:
        video_model = paddle.DataParallel(video_model)

    pre_state_dict = paddle.load(args.pretrain)
    #if paddle.distributed.parallel.Env().local_rank == 0:
    video_model = init_model(video_model, pre_state_dict)

    optimizer = create_optimizer(train_config.TRAIN, video_model.parameters())

    bs_denominator = 1
    if args.use_gpu:
        # check number of GPUs
        gpus = os.getenv("CUDA_VISIBLE_DEVICES", "")
        if gpus == "":
            pass
        else:
            gpus = gpus.split(",")
            num_gpus = len(gpus)
        bs_denominator = num_gpus
    bs_train_single = int(train_config.TRAIN.batch_size / bs_denominator)
    bs_val_single = int(valid_config.VALID.batch_size / bs_denominator)

    train_dataset = TSN_UCF101_Dataset(train_config, 'train')
    val_dataset = TSN_UCF101_Dataset(valid_config, 'valid')
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

    # resume training the model
    if args.resume is not None:
        model_state, opt_state = paddle.load(args.resume)
        video_model.set_dict(model_state)
        optimizer.set_dict(opt_state)

    reader_cost_averager = TimeAverager()
    batch_cost_averager = TimeAverager()
    for epoch in range(1, train_config.TRAIN.epoch + 1):
        epoch_start = time.time()

        video_model.train()
        total_loss = 0.0
        total_acc1 = 0.0
        total_acc5 = 0.0
        total_sample = 0

        batch_start = time.time()
        for batch_id, data in enumerate(train_loader):
            reader_cost_averager.record(time.time() - batch_start)

            imgs = paddle.to_tensor(data[0], place=paddle.CUDAPinnedPlace())
            labels = paddle.to_tensor(data[1], place=paddle.CUDAPinnedPlace())
            labels.stop_gradient = True
            outputs = video_model(imgs)

            loss = F.cross_entropy(input=outputs, label=labels, ignore_index=-1)
            avg_loss = paddle.mean(loss)

            acc_top1 = paddle.metric.accuracy(input=outputs, label=labels, k=1)
            acc_top5 = paddle.metric.accuracy(input=outputs, label=labels, k=5)
            dy_out = avg_loss.numpy()[0]

            if use_data_parallel:
                # (data_parallel step5/6)
                avg_loss = video_model.scale_loss(avg_loss)
                avg_loss.backward()
                video_model.apply_collective_grads()
            else:
                avg_loss.backward()

            optimizer.minimize(avg_loss)
            optimizer.step()
            optimizer.clear_grad()

            total_loss += dy_out
            total_acc1 += acc_top1.numpy()[0]
            total_acc5 += acc_top5.numpy()[0]
            total_sample += 1

            batch_cost_averager.record(
                time.time() - batch_start, num_samples=bs_train_single)
            if batch_id % args.log_interval == 0:
                print(
                    'TRAIN Epoch: {}, iter: {}, loss={:.6f}, acc1 {:.6f}, acc5 {:.6f}, batch_cost: {:.5f} sec, reader_cost: {:.5f} sec, ips: {:.5f} samples/sec'.
                    format(epoch, batch_id, total_loss / total_sample,
                           total_acc1 / total_sample, total_acc5 / total_sample,
                           batch_cost_averager.get_average(),
                           reader_cost_averager.get_average(),
                           batch_cost_averager.get_ips_average()))
                batch_cost_averager.reset()
                reader_cost_averager.reset()

            batch_start = time.time()

        train_epoch_cost = time.time() - epoch_start
        print(
            'TRAIN End, Epoch {}, avg_loss= {:.6f}, avg_acc1= {:.6f}, avg_acc5= {:.6f}, epoch_cost: {:.5f} sec'.
            format(epoch, total_loss / total_sample, total_acc1 / total_sample,
                   total_acc5 / total_sample, train_epoch_cost))

        # save model's and optimizer's parameters which used for resuming the training stage
        save_parameters = (not use_data_parallel) or (
            use_data_parallel and
            paddle.distributed.ParallelEnv().local_rank == 0)
        if save_parameters:
            model_path_pre = "_tsn"
            if not os.path.isdir(args.checkpoint):
                os.makedirs(args.checkpoint)
            model_path = os.path.join(
                args.checkpoint,
                "_" + model_path_pre + "_epoch{}".format(epoch))
            paddle.save(video_model.state_dict(), model_path)
            paddle.save(optimizer.state_dict(), model_path)

        if args.validate:
            video_model.eval()
            val_acc = val(epoch, video_model, val_loader, valid_config, args)
            # save the best parameters in trainging stage
            if epoch == 1:
                best_acc = val_acc
            else:
                if val_acc > best_acc:
                    best_acc = val_acc
                    if paddle.distributed.ParallelEnv().local_rank == 0:
                        if not os.path.isdir(args.weights):
                            os.makedirs(args.weights)
                        paddle.save(video_model.state_dict(),
                                    args.weights + "/final")
        else:
            if paddle.distributed.parallel.Env().local_rank == 0:
                if not os.path.isdir(args.weights):
                    os.makedirs(args.weights)
                paddle.save(video_model.state_dict(), args.weights + "/final")

    logger.info('[TRAIN] training finished')


if __name__ == "__main__":
    args = parse_args()
    logger.info(args)
    train(args)
