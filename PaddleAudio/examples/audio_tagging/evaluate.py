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
import paddle.nn.functional as F
import paddleaudio as pa
import yaml
from dataset import get_loader
from model import *
from paddle.utils import download
from sklearn.metrics import average_precision_score
from utils import (get_label_name_mapping, get_labels527, get_logger, get_metrics)

with open('./config.yaml') as f:
    c = yaml.safe_load(f)

logger = get_logger(__name__, os.path.join(c['log_path'], 'inference.txt'))

checkpoint_url = 'https://bj.bcebos.com/paddleaudio/paddleaudio/mixup_resnet50_checkpoint33.pdparams'


def evaluate(epoch, val_loader, model, loss_fn, log_writer=None):
    model.eval()
    avg_loss = 0.0
    avg_preci = 0.0
    avg_recall = 0.0
    all_labels = []
    all_preds = []
    for batch_id, data in enumerate(val_loader()):
        xd, yd = data
        xd = xd.unsqueeze((1))
        label = yd
        logits = model(xd)
        loss_val = loss_fn(logits, label)

        pred = F.softmax(logits)
        all_labels += [label.numpy()]
        all_preds += [pred.numpy()]

        preci, recall = get_metrics(label, pred)
        avg_preci = (avg_preci * batch_id + preci) / (1 + batch_id)
        avg_recall = (avg_recall * batch_id + recall) / (1 + batch_id)
        avg_loss = (avg_loss * batch_id + loss_val.numpy()[0]) / (1 + batch_id)

        msg = f'eval epoch:{epoch}, batch:{batch_id}'
        msg += f'|{len(val_loader)}'
        msg += f',loss:{avg_loss:.3}'
        msg += f',recall:{avg_recall:.3}'
        msg += f',preci:{avg_preci:.3}'
        avg_preci = (avg_preci * batch_id + preci) / (1 + batch_id)
        avg_recall = (avg_recall * batch_id + recall) / (1 + batch_id)
        if batch_id % 20 == 0:
            logger.info(msg)
            if log_writer is not None:
                log_writer.add_scalar(tag="eval loss", step=batch_id, value=avg_loss)
                log_writer.add_scalar(tag="eval preci", step=batch_id, value=avg_preci)
                log_writer.add_scalar(tag="eval recall", step=batch_id, value=avg_recall)

    all_preds = np.concatenate(all_preds, 0)
    all_labels = np.concatenate(all_labels, 0)
    mAP_scores = average_precision_score(all_labels, all_preds, average=None)

    return avg_loss, avg_preci, avg_recall, mAP_scores


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Audioset inference')
    parser.add_argument('--device', help='set the gpu device number', type=int, required=False, default=0)
    parser.add_argument('--weight', type=str, required=False, default='')
    args = parser.parse_args([])

    paddle.set_device('gpu:{}'.format(args.device))
    ModelClass = eval(c['model_type'])
    model = ModelClass(pretrained=False, num_classes=c['num_classes'], dropout=c['dropout'])

    if args.weight.strip() == '':
        args.weight = download.get_weights_path_from_url(checkpoint_url)
    model.load_dict(paddle.load(args.weight))
    model.eval()

    _, val_loader = get_loader()
    logger.info(f'evaluating...')

    val_acc, val_preci, val_recall, mAP_scores = evaluate(0, val_loader, model, F.binary_cross_entropy_with_logits)
    avg_map = np.mean(mAP_scores)
    logger.info(f'average mAP: {avg_map}')
