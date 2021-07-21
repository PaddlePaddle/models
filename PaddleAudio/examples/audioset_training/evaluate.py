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
from dataset import get_val_loader
from model import resnet50
from paddle.utils import download
from sklearn.metrics import average_precision_score, roc_auc_score
from utils import compute_dprime, download_assets

checkpoint_url = 'https://bj.bcebos.com/paddleaudio/paddleaudio/resnet50_weight_averaging_mAP0.416.pdparams'


def evaluate(epoch, val_loader, model, loss_fn):
    model.eval()
    avg_loss = 0.0
    all_labels = []
    all_preds = []
    for batch_id, (x, y) in enumerate(val_loader()):
        x = x.unsqueeze((1))
        label = y
        logits = model(x)
        loss_val = loss_fn(logits, label)

        pred = F.sigmoid(logits)
        all_labels += [label.numpy()]
        all_preds += [pred.numpy()]
        avg_loss = (avg_loss * batch_id + loss_val.numpy()[0]) / (1 + batch_id)
        msg = f'eval epoch:{epoch}, batch:{batch_id}'
        msg += f'|{len(val_loader)}'
        msg += f',loss:{avg_loss:.3}'
        if batch_id % 20 == 0:
            print(msg)

    all_preds = np.concatenate(all_preds, 0)
    all_labels = np.concatenate(all_labels, 0)
    mAP_score = np.mean(
        average_precision_score(all_labels, all_preds, average=None))
    auc_score = np.mean(roc_auc_score(all_labels, all_preds, average=None))
    dprime = compute_dprime(auc_score)
    return avg_loss, mAP_score, auc_score, dprime


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Audioset inference')
    parser.add_argument('--config',
                        type=str,
                        required=False,
                        default='./assets/config.yaml')
    parser.add_argument('--device',
                        help='set the gpu device number',
                        type=int,
                        required=False,
                        default=0)
    parser.add_argument('--weight', type=str, required=False, default='')
    args = parser.parse_args()
    download_assets()
    with open(args.config) as f:
        c = yaml.safe_load(f)
    paddle.set_device('gpu:{}'.format(args.device))
    ModelClass = eval(c['model_type'])
    model = ModelClass(pretrained=False,
                       num_classes=c['num_classes'],
                       dropout=c['dropout'])
    if args.weight.strip() == '':
        print(f'Using pretrained weight: {checkpoint_url}')
        args.weight = download.get_weights_path_from_url(checkpoint_url)
    model.load_dict(paddle.load(args.weight))
    model.eval()

    val_loader = get_val_loader(c)

    print(f'Evaluating...')
    avg_loss, mAP_score, auc_score, dprime = evaluate(
        0, val_loader, model, F.binary_cross_entropy_with_logits)

    print(f'mAP: {mAP_score:.3}')
    print(f'auc: {auc_score:.3}')
    print(f'd-prime: {dprime:.3}')
