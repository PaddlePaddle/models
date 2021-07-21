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
import paddle.nn.functional as F
import yaml
from dataset import get_train_loader, get_val_loader
from evaluate import evaluate
from model import resnet18, resnet50, resnet101
from paddle.io import DataLoader, Dataset, IterableDataset
from paddle.optimizer import Adam
from utils import (MixUpLoss, get_metrics, load_checkpoint, mixup_data,
                   save_checkpoint)
from visualdl import LogWriter

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Audioset training')
    parser.add_argument('--device', type=int, required=False, default=1)
    parser.add_argument('--restore', type=int, required=False, default=-1)
    parser.add_argument('--config',
                        type=str,
                        required=False,
                        default='./assets/config.yaml')
    parser.add_argument('--distributed', type=int, required=False, default=0)
    args = parser.parse_args()
    with open(args.config) as f:
        c = yaml.safe_load(f)
    log_writer = LogWriter(logdir=c['log_path'])

    prefix = 'mixup_{}'.format(c['model_type'])
    if args.distributed != 0:
        dist.init_parallel_env()
        local_rank = dist.get_rank()
    else:
        paddle.set_device('gpu:{}'.format(args.device))
        local_rank = 0

    print(f'using ' + c['model_type'])
    ModelClass = eval(c['model_type'])

    #define loss
    bce_loss = F.binary_cross_entropy_with_logits
    loss_fn = MixUpLoss(bce_loss)

    warm_steps = c['warm_steps']
    lrs = np.linspace(1e-10, c['start_lr'], warm_steps)

    # restore checkpoint
    if args.restore != -1:
        model = ModelClass(pretrained=False,
                           num_classes=c['num_classes'],
                           dropout=c['dropout'])
        model_dict, optim_dict = load_checkpoint(c['model_dir'], args.restore,
                                                 prefix)
        model.load_dict(model_dict)
        optimizer = Adam(learning_rate=c['start_lr'],
                         parameters=model.parameters())
        optimizer.set_state_dict(optim_dict)
        start_epoch = args.restore

    else:
        model = ModelClass(pretrained=True,
                           num_classes=c['num_classes'],
                           dropout=c['dropout'])  # use imagenet pretrained
        optimizer = Adam(learning_rate=c['start_lr'],
                         parameters=model.parameters())
        start_epoch = 0

    #for name,p in list(model.named_parameters())[:-2]:
    # print(name,p.stop_gradient)
    # p.stop_gradient = True

    os.makedirs(c['model_dir'], exist_ok=True)
    if args.distributed != 0:
        model = paddle.DataParallel(model)

    train_loader = get_train_loader(c)
    val_loader = get_val_loader(c)

    epoch_num = c['epoch_num']
    if args.restore != -1:

        avg_loss, mAP_score, auc_score, dprime = evaluate(
            args.restore, val_loader, model, bce_loss)
        print(f'average map at epoch {args.restore} is {mAP_score}')
        print(f'auc_score: {auc_score}')
        print(f'd-prime: {dprime}')

        best_mAP = mAP_score

        log_writer.add_scalar(tag="eval mAP",
                              step=args.restore,
                              value=mAP_score)
        log_writer.add_scalar(tag="eval auc",
                              step=args.restore,
                              value=auc_score)
        log_writer.add_scalar(tag="eval dprime",
                              step=args.restore,
                              value=dprime)
    else:
        best_mAP = 0.0

    step = 0
    for epoch in range(start_epoch, epoch_num):

        avg_loss = 0.0
        avg_preci = 0.0
        avg_recall = 0.0

        model.train()
        model.clear_gradients()
        t0 = time.time()
        for batch_id, (x, y) in enumerate(train_loader()):
            if step < warm_steps:
                optimizer.set_lr(lrs[step])
            x.stop_gradient = False
            if c['balanced_sampling']:
                x = x.squeeze()
                y = y.squeeze()
            x = x.unsqueeze((1))
            if c['mixup']:
                mixed_x, mixed_y = mixup_data(x, y, c['mixup_alpha'])
                logits = model(mixed_x)
                loss_val = loss_fn(logits, mixed_y)
                loss_val.backward()
            else:
                logits = model(x)
                loss_val = bce_loss(logits, y)
                loss_val.backward()
            optimizer.step()
            model.clear_gradients()
            pred = F.sigmoid(logits)
            preci, recall = get_metrics(y.squeeze().numpy(), pred.numpy())
            avg_loss = (avg_loss * batch_id + loss_val.numpy()[0]) / (1 +
                                                                      batch_id)
            avg_preci = (avg_preci * batch_id + preci) / (1 + batch_id)
            avg_recall = (avg_recall * batch_id + recall) / (1 + batch_id)

            elapsed = (time.time() - t0) / 3600
            remain = elapsed / (1 + batch_id) * (len(train_loader) - batch_id)

            msg = f'epoch:{epoch}, batch:{batch_id}'
            msg += f'|{len(train_loader)}'
            msg += f',loss:{avg_loss:.3}'
            msg += f',recall:{avg_recall:.3}'
            msg += f',preci:{avg_preci:.3}'
            msg += f',elapsed:{elapsed:.1}h'
            msg += f',remained:{remain:.1}h'
            if batch_id % 20 == 0 and local_rank == 0:
                print(msg)
                log_writer.add_scalar(tag="train loss",
                                      step=step,
                                      value=avg_loss)
                log_writer.add_scalar(tag="train preci",
                                      step=step,
                                      value=avg_preci)
                log_writer.add_scalar(tag="train recall",
                                      step=step,
                                      value=avg_recall)

            step += 1
            if step % c['checkpoint_step'] == 0 and local_rank == 0:
                save_checkpoint(c['model_dir'], epoch, model, optimizer, prefix)

                avg_loss, avg_map, auc_score, dprime = evaluate(
                    epoch, val_loader, model, bce_loss)
                print(f'average map at epoch {epoch} is {avg_map}')
                print(f'auc: {auc_score}')
                print(f'd-prime: {dprime}')

                log_writer.add_scalar(tag="eval mAP", step=epoch, value=avg_map)
                log_writer.add_scalar(tag="eval auc",
                                      step=epoch,
                                      value=auc_score)
                log_writer.add_scalar(tag="eval dprime",
                                      step=epoch,
                                      value=dprime)

                model.train()
                model.clear_gradients()

                if avg_map > best_mAP:
                    print('mAP improved from {} to {}'.format(
                        best_mAP, avg_map))
                    best_mAP = avg_map
                    fn = os.path.join(
                        c['model_dir'],
                        f'{prefix}_epoch{epoch}_mAP{avg_map:.3}.pdparams')
                    paddle.save(model.state_dict(), fn)
                else:
                    print(f'mAP {avg_map} did not improved from {best_mAP}')

            if step % c['lr_dec_per_step'] == 0 and step != 0:
                if optimizer.get_lr() <= 1e-6:
                    factor = 0.95
                else:
                    factor = 0.8
                optimizer.set_lr(optimizer.get_lr() * factor)
                print('decreased lr to {}'.format(optimizer.get_lr()))
