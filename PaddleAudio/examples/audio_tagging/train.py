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

import os
import time
import glob
import numpy as np
import paddle
import argparse
from dataset import get_loader
from mixup_utils import mixup_data, MixUpLoss
from model import *
import paddle.distributed as dist
from paddle.io import Dataset, DataLoader, IterableDataset
import paddle.nn.functional as F
from paddle.optimizer import Adam
from utils import get_logger, get_metrics
from utils import load_checkpoint, save_checkpoint
from evaluate import evaluate
from visualdl import LogWriter
import yaml

with open('./config.yaml') as f:
    c = yaml.safe_load(f)

log_writer = LogWriter(logdir=c['log_path'])
logger = get_logger(__name__, os.path.join(c['log_path'], 'log.txt'))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Audioset training')
    parser.add_argument('--device', type=int, required=False, default=0)
    parser.add_argument('--restore', type=int, required=False, default=-1)
    parser.add_argument('--distributed', type=int, required=False, default=0)

    last_model = None

    args = parser.parse_args()
    prefix = 'mixup_{}'.format(c['model_type'])

    if args.distributed != 0:
        dist.init_parallel_env()
        local_rank = dist.get_rank()
    else:
        paddle.set_device('gpu:{}'.format(args.device))
        local_rank = 0

    logger.info(f'using ' + c['model_type'])
    ModelClass = eval(c['model_type'])

    #define loss    
    bce_loss = F.binary_cross_entropy_with_logits
    loss_fn = MixUpLoss(bce_loss)

    # restore checkpoint
    if args.restore != -1:
        model = ModelClass(pretrained=False, num_classes=c['num_classes'], dropout=c['dropout'])
        model_dict, optim_dict = load_checkpoint(args.restore, prefix)
        model.load_dict(model_dict)
        # when loading a state dict, stop_gradent must set to False manully
        # for p in model.parameters():
        # print(p.stop_gradient)
        # p.stop_gradient = False
        optimizer = Adam(learning_rate=c['start_lr'], parameters=model.parameters())
        optimizer.set_state_dict(optim_dict)
        start_poch = args.restore

    else:
        model = ModelClass(
            pretrained=True, num_classes=c['num_classes'], dropout=c['dropout'])  # use imagenet pretrained 
        optimizer = Adam(learning_rate=c['start_lr'], parameters=model.parameters())
        start_epoch = 0

    os.makedirs(c['model_dir'], exist_ok=True)
    if args.distributed != 0:
        model = paddle.DataParallel(model)

    train_loader, val_loader = get_loader()

    epoch_num = c['epoch_num']
    if args.restore != -1:
        val_acc, val_preci, val_recall, mAP_scores = evaluate(args.restore, val_loader, model, bce_loss, log_writer)
        avg_map = np.mean(mAP_scores)
        logger.info(f'average at epoch {args.restore} is {avg_map}')
    step = 0
    best_acc = 0
    best_mAP = 0
    for epoch in range(start_poch, epoch_num):

        train_loader.dataset.shuffle()
        avg_loss = 0.0
        avg_preci = 0.0
        avg_recall = 0.0

        model.train()
        model.clear_gradients()
        t0 = time.time()
        for batch_id, data in enumerate(train_loader()):
            xd, yd = data
            xd.stop_gradient = False
            yd.stop_gradient = False
            if c['balance_sampling']:
                xd = xd.squeeze()
                yd = yd.squeeze()
            xd = xd.unsqueeze((1))
            if c['mixup']:
                mixed_x, mixed_y = mixup_data(xd, yd, c['mixup_alpha'])
                logits = model(mixed_x)
                loss_val = loss_fn(logits, mixed_y)
                loss_val.backward()
            else:
                logits = model(xd)
                loss_val = bce_loss(logits, yd)
                loss_val.backward()
            optimizer.step()
            model.clear_gradients()
            pred = F.softmax(logits)
            preci, recall = get_metrics(yd.squeeze(), pred)
            avg_loss = (avg_loss * batch_id + loss_val.numpy()[0]) / (1 + batch_id)
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
                logger.info(msg)
                log_writer.add_scalar(tag="train loss", step=step, value=avg_loss)
                log_writer.add_scalar(tag="train preci", step=step, value=avg_preci)
                log_writer.add_scalar(tag="train recall", step=step, value=avg_recall)

            step += 1
            if step % c['checkpoint_step'] == 0 and local_rank == 0:
                save_checkpoint(epoch, model, optimizer, prefix)

                val_acc, val_preci, val_recall, mAP_scores = evaluate(epoch, val_loader, model, bce_loss, log_writer)
                avg_map = np.mean(mAP_scores)

                if avg_map > best_mAP:
                    logger.info('mAP improved from {} to {}'.format(best_mAP, avg_map))
                    best_mAP = avg_map
                    if last_model is not None:
                        os.remove(last_model)

                    fn = os.path.join(c['model_dir'], '{}_epoch{}_mAP{:.3}_preci{:.3}_recall{:.3}.pdparams'.format(
                        prefix, epoch, avg_map, val_preci, val_recall))
                    paddle.save(model.state_dict(), fn)
                    last_model = fn
                else:
                    logger.info(f'mAP {avg_map} did not improved from {best_mAP}')

            if step % c['lr_dec_per_step'] == 0 and step != 0:
                if optimizer.get_lr() <= 3e-6:
                    factor = 0.95
                else:
                    factor = 0.1
                optimizer.set_lr(optimizer.get_lr() * factor)
                logger.info('decreased lr to {}'.format(optimizer.get_lr()))
