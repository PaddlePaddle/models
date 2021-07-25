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
from model import resnet50
from paddle.utils import download
from paddleaudio.functional import melspectrogram
from utils import (download_assets, get_label_name_mapping, get_labels,
                   get_metrics)

download_assets()

checkpoint_url = 'https://bj.bcebos.com/paddleaudio/paddleaudio/resnet50_weight_averaging_mAP0.416.pdparams'


def load_and_extract_feature(file, c):
    s, _ = pa.load(file, sr=c['sample_rate'])
    x = melspectrogram(paddle.to_tensor(s),
                       sr=c['sample_rate'],
                       win_length=c['window_size'],
                       n_fft=c['window_size'],
                       hop_length=c['hop_size'],
                       n_mels=c['mel_bins'],
                       f_min=c['fmin'],
                       f_max=c['fmax'],
                       window='hann',
                       center=True,
                       pad_mode='reflect',
                       to_db=True,
                       amin=1e-3,
                       top_db=None)
    x = x.transpose((0, 2, 1))
    x = x.unsqueeze((0, ))
    return x


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Audioset inference')

    parser.add_argument('--device',
                        help='set the gpu device number',
                        type=int,
                        required=False,
                        default=0)
    parser.add_argument('--config',
                        type=str,
                        required=False,
                        default='./assets/config.yaml')
    parser.add_argument('--weight', type=str, required=False, default='')
    parser.add_argument('--wav_file',
                        type=str,
                        required=False,
                        default='./wav/TKtNAJa-mbQ_11.000.wav')

    parser.add_argument('--top_k', type=int, required=False, default=20)
    args = parser.parse_args()
    top_k = args.top_k
    label2name, name2label = get_label_name_mapping()

    with open(args.config) as f:
        c = yaml.safe_load(f)

    paddle.set_device('gpu:{}'.format(args.device))
    ModelClass = eval(c['model_type'])
    model = ModelClass(pretrained=False,
                       num_classes=c['num_classes'],
                       dropout=c['dropout'])

    if args.weight.strip() == '':
        args.weight = download.get_weights_path_from_url(checkpoint_url)

    model.load_dict(paddle.load(args.weight))
    model.eval()
    x = load_and_extract_feature(args.wav_file, c)
    labels = get_labels()
    logits = model(x)
    pred = F.sigmoid(logits)
    pred = pred[0].cpu().numpy()

    clsidx = np.argsort(pred)[-top_k:][::-1]
    probs = np.sort(pred)[-top_k:][::-1]
    for i, idx in enumerate(clsidx):
        name = label2name[labels[idx]]
        print(name, probs[i])
