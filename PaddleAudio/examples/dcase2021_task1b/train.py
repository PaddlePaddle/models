# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved
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

import argparse
import glob
import os
import time

import numpy as np
import paddle
import paddle.distributed as dist
import paddle.nn as nn
import paddle.nn.functional as F
import yaml
from dataset import get_train_loader, get_val_loader
from evaluate import evaluate
from model import resnet18, resnet50, resnet101
from paddle.io import DataLoader, Dataset, IterableDataset
from paddle.optimizer import Adam
from paddle.utils import download
from paddleaudio.utils.log import logger
from utils import MixUpLoss, load_checkpoint, mixup_data, save_checkpoint
from visualdl import LogWriter

AUDIOSET_URL = 'https://bj.bcebos.com/paddleaudio/examples/audioset/weights/resnet50_map0.416.pdparams'

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Audioset training')
    parser.add_argument(
        '--device',
        choices=['cpu', 'gpu'],
        default="gpu",
        help="Select which device to train model, defaults to gpu.")
    parser.add_argument('--restore', type=int, required=False, default=-1)
    parser.add_argument(
        '--task_type',
        choices=['audio-only', 'audio_visual'],
        help='task type',
        type=str,
        required=False,
        default='audio_only')
    parser.add_argument(
        '--config', type=str, required=False, default='./assets/config.yaml')
    parser.add_argument('--distributed', type=int, required=False, default=0)

    args = parser.parse_args()
    with open(args.config) as f:
        c = yaml.safe_load(f)
    log_writer = LogWriter(logdir=c['log_path'])
    mixup = c['mixup']
    prefix = f'dcase21task1b_resnet50_{args.task_type}_mixup={mixup}'

    if args.distributed != 0:
        dist.init_parallel_env()
        local_rank = dist.get_rank()
    else:
        paddle.set_device(args.device)
        local_rank = 0

    logger.info(f'using ' + c['model_type'])
    ModelClass = eval(c['model_type'])
    #define loss and lr
    nll_loss = nn.NLLLoss()
    mixup_loss = MixUpLoss(nll_loss)
    warm_steps = c['warm_steps']
    lrs = np.linspace(1e-10, c['start_lr'], warm_steps)
    # restore checkpoint
    if args.restore != -1:
        model = ModelClass(
            pretrained=False,
            num_classes=c['num_classes'],
            task_type=args.task_type,
            dropout=c['dropout'])
        model_dict, optim_dict = load_checkpoint(c['model_dir'], args.restore,
                                                 prefix)
        model.load_dict(model_dict)
        optimizer = Adam(
            learning_rate=c['start_lr'], parameters=model.parameters())
        optimizer.set_state_dict(optim_dict)
        start_epoch = args.restore
    else:
        model = ModelClass(
            pretrained=False,
            num_classes=c['num_classes'],
            task_type=args.task_type,
            dropout=c['dropout'])  # use imagenet pretrained

        start_epoch = 0
        logger.info(f'Using pretrained weight: {AUDIOSET_URL}')
        weight = download.get_weights_path_from_url(AUDIOSET_URL)
        model.load_dict(paddle.load(weight))
        optimizer = Adam(
            learning_rate=c['start_lr'], parameters=model.parameters())

    os.makedirs(c['model_dir'], exist_ok=True)
    if args.distributed != 0:
        model = paddle.DataParallel(model)

    train_loader = get_train_loader(c)
    val_loader = get_val_loader(c)

    epoch_num = c['epoch_num']
    if args.restore != -1:
        avg_loss, val_acc = evaluate(args.restore, val_loader, model, nll_loss,
                                     args.task_type)
        best_acc = val_acc
        log_writer.add_scalar(
            tag="eval loss", step=args.restore, value=avg_loss)
        log_writer.add_scalar(tag="eval acc", step=args.restore, value=val_acc)
    else:
        best_acc = 0.0

    step = 0
    for epoch in range(start_epoch, epoch_num):
        avg_loss = 0.0
        avg_acc = 0.0
        model.train()
        model.clear_gradients()
        t0 = time.time()
        for batch_id, (xd, yd, p) in enumerate(train_loader()):
            if args.task_type == 'audio_only':
                p = None  # do not use probability from video branch
            if step < warm_steps:
                optimizer.set_lr(lrs[step])
            xd.stop_gradient = False
            if c['balanced_sampling']:
                xd = xd.squeeze()
                yd = yd.squeeze()
            xd = xd.unsqueeze((1))
            if c['mixup']:
                mixed_x, mixed_p, mixed_y = mixup_data(xd, yd, p,
                                                       c['mixup_alpha'])
                logits = model(mixed_x, mixed_p)
                pred = F.log_softmax(logits)
                loss_val = mixup_loss(pred, mixed_y)
                loss_val.backward()
            else:
                logits = model(xd, p)
                pred = F.log_softmax(logits)
                loss_val = nll_loss(pred, yd)
                loss_val.backward()
            optimizer.step()
            optimizer.clear_grad()

            acc = np.mean(np.argmax(pred.numpy(), axis=1) == yd.numpy())
            avg_loss = (avg_loss * batch_id + loss_val.numpy()[0]) / (
                1 + batch_id)
            avg_acc = (avg_acc * batch_id + acc) / (1 + batch_id)
            elapsed = (time.time() - t0) / 3600
            remain = elapsed / (1 + batch_id) * (len(train_loader) - batch_id)

            msg = f'epoch:{epoch}, batch:{batch_id}'
            msg += f'|{len(train_loader)}'
            msg += f',loss:{avg_loss:.3}'
            msg += f',acc:{avg_acc:.3}'
            msg += f',lr:{optimizer.get_lr():.2}'
            msg += f',elapsed:{elapsed:.1}h'
            msg += f',remained:{remain:.1}h'

            if batch_id % 50 == 0 and local_rank == 0:
                logger.info(msg)
                log_writer.add_scalar(
                    tag="train loss", step=step, value=avg_loss)
                log_writer.add_scalar(tag="train acc", step=step, value=avg_acc)
            step += 1
            if step % c['checkpoint_step'] == 0 and local_rank == 0:

                val_loss, val_acc = evaluate(epoch, val_loader, model, nll_loss,
                                             args.task_type)
                log_writer.add_scalar(tag="eval acc", step=epoch, value=val_acc)
                log_writer.add_scalar(
                    tag="eval loss", step=epoch, value=val_loss)
                model.train()
                model.clear_gradients()

                if val_acc > best_acc:
                    logger.info('acc improved from {} to {}'.format(best_acc,
                                                                    val_acc))
                    best_acc = val_acc
                    fn = os.path.join(
                        c['model_dir'],
                        f'{prefix}_epoch{epoch}_acc{val_acc:.3}.pdparams')
                    paddle.save(model.state_dict(), fn)
                else:
                    logger.info(
                        f'acc {val_acc} did not improved from {best_acc}')

            if step % c['lr_dec_per_step'] == 0 and step != 0:
                if optimizer.get_lr() <= 1e-6:
                    factor = 0.95
                else:
                    factor = 0.8
                optimizer.set_lr(optimizer.get_lr() * factor)
                logger.info('decreased lr to {}'.format(optimizer.get_lr()))
