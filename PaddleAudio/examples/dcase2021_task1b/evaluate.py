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
import os

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddleaudio as pa
import yaml
from dataset import get_val_loader
from model import resnet50
from paddle.utils import download
from paddleaudio.utils.log import logger

checkpoint_url = {
    'audio_only':
    'https://bj.bcebos.com/paddleaudio/examples/dcase21_task1b/weights/r50_audio_only.pdparams',
    'audio_visual':
    'https://bj.bcebos.com/paddleaudio/examples/dcase21_task1b/weights/r50_audio_visual.pdparams'
}


def evaluate(epoch, val_loader, model, loss_fn, task_type='audio_only'):
    model.eval()
    avg_loss = 0.0
    avg_acc = 0.0
    for batch_id, (x, y, p) in enumerate(val_loader()):
        x = x.unsqueeze((1))
        label = y
        if task_type == 'audio_only':
            logits = model(x)
        else:
            logits = model(x, p)

        pred = F.log_softmax(logits)
        loss_val = loss_fn(pred, label)
        acc = np.mean(np.argmax(pred.numpy(), axis=1) == y.numpy())
        avg_loss = (avg_loss * batch_id + loss_val.numpy()[0]) / (1 + batch_id)
        avg_acc = (avg_acc * batch_id + acc) / (1 + batch_id)

        msg = f'eval epoch:{epoch}, batch:{batch_id}'
        msg += f'|{len(val_loader)}'
        msg += f',loss:{avg_loss:.3}'
        msg += f',acc:{avg_acc:.3}'

        if batch_id % 20 == 0:
            logger.info(msg)

    return avg_loss, avg_acc


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='dcase 2021 task1b')
    parser.add_argument(
        '--config', type=str, required=False, default='./assets/config.yaml')
    parser.add_argument(
        '--device',
        choices=['cpu', 'gpu'],
        default="gpu",
        help="Select which device to train model, defaults to gpu.")
    parser.add_argument(
        '--task_type',
        choices=['audio-only', 'audio_visual'],
        help='task type',
        type=str,
        required=False,
        default='audio_only')  # or audio_visual
    parser.add_argument('--weight', type=str, required=False, default='')
    args = parser.parse_args()

    assert args.task_type in [
        'audio_only', 'auido_visual'
    ], 'task_type must be one of [audio_only auido_visual]'
    with open(args.config) as f:
        c = yaml.safe_load(f)
    paddle.set_device(args.device)
    ModelClass = eval(c['model_type'])
    model = ModelClass(
        pretrained=False,
        num_classes=c['num_classes'],
        dropout=c['dropout'],
        task_type=args.task_type)
    if args.weight.strip() == '':
        logger.info(
            f'Using pretrained weight: {checkpoint_url[args.task_type]}')
        args.weight = download.get_weights_path_from_url(checkpoint_url[
            args.task_type])
    model.load_dict(paddle.load(args.weight))
    model.eval()

    val_loader = get_val_loader(c)

    logger.info(f'Evaluating...')
    val_loss, val_acc = evaluate(
        0, val_loader, model, nn.NLLLoss(), task_type=args.task_type)
    logger.info(f'Overall acc: {val_acc:.3}')
    logger.info(f'Overall loss: {val_loss:.3}')
