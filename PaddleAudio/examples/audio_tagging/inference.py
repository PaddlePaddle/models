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
import numpy as np
import paddle
import argparse
from model import *
import paddle.nn.functional as F
import paddleaudio as pa
from utils import get_logger, get_metrics
from utils import get_label_name_mapping
from utils import get_labels527
import yaml
from paddle.utils import download

with open('./config.yaml') as f:
    c = yaml.safe_load(f)
checkpoint_url = 'https://bj.bcebos.com/paddleaudio/paddleaudio/mixup_resnet50_checkpoint33.pdparams'
logger = get_logger(__name__, os.path.join(c['log_path'], 'inference.txt'))


def load_and_extract_feature(file):
    s, r = pa.load(file, sr=c['sample_rate'])
    x = pa.features.mel_spect(
        s,
        sample_rate=c['sample_rate'],
        window_size=c['window_size'],
        hop_length=c['hop_size'],
        mel_bins=c['mel_bins'],
        fmin=c['fmin'],
        fmax=c['fmax'],
        window='hann',
        center=True,
        pad_mode='reflect',
        ref=1.0,
        amin=1e-10,
        top_db=None)

    x = x.T  #!!
    x = paddle.Tensor(x).unsqueeze((0, 1))
    return x


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Audioset inference')
    parser.add_argument('--device', help='set the gpu device number', type=int, required=False, default=0)
    parser.add_argument('--weight', type=str, required=False, default='')
    parser.add_argument('--wav_file', type=str, required=False, default='./wav/TKtNAJa-mbQ_11.000.wav')
    parser.add_argument('--top_k', type=int, required=False, default=5)
    args = parser.parse_args([])
    top_k = args.top_k
    label2name, name2label = get_label_name_mapping()
    paddle.set_device('gpu:{}'.format(args.device))
    ModelClass = eval(c['model_type'])
    model = ModelClass(pretrained=False, num_classes=c['num_classes'], dropout=c['dropout'])

    if args.weight.strip() == '':
        args.weight = download.get_weights_path_from_url(checkpoint_url)

    model.load_dict(paddle.load(args.weight))
    model.eval()
    x = load_and_extract_feature(args.wav_file)
    labels = get_labels527()
    logits = model(x)
    pred = F.sigmoid(logits)
    pred = pred[0].cpu().numpy()

    clsidx = np.argsort(pred)[-top_k:][::-1]
    probs = np.sort(pred)[-top_k:][::-1]
    for i, idx in enumerate(clsidx):
        name = label2name[labels[idx]]
        print(name, probs[i])
