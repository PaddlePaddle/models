# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import ast
import os

import numpy as np
import paddle
import paddle.nn.functional as F
from model import SoundClassifier
from paddleaudio.backends import load as load_audio
from paddleaudio.datasets import ESC50
from paddleaudio.features import mel_spect
from paddleaudio.models.panns import cnn14

parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--use_gpu",
                    type=ast.literal_eval,
                    default=True,
                    help="Whether use GPU for predicting. Input should be True or False")
parser.add_argument("--wav", type=str, required=True, help="Audio file to infer.")
parser.add_argument("--sr", type=int, default=44100, help="Sample rate of inference audio.")
parser.add_argument("--topk", type=int, default=1, help="Show top k results of prediction labels.")
parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint of model.")
args = parser.parse_args()


def extract_features(file: str, **kwargs):
    waveform = load_audio(args.wav, sr=args.sr)[0]
    feats = mel_spect(waveform, sample_rate=args.sr, **kwargs).transpose()
    return feats


if __name__ == '__main__':
    paddle.set_device('gpu') if args.use_gpu else paddle.set_device('cpu')

    model = SoundClassifier(model=cnn14(pretrained=False, extract_embedding=True), num_class=len(ESC50.label_list))
    model.set_state_dict(paddle.load(args.checkpoint))
    model.eval()

    feats = extract_features(args.wav)
    feats = paddle.to_tensor(np.expand_dims(feats, 0))
    logits = model(feats)
    probs = F.softmax(logits, axis=1).numpy()

    sorted_indices = (-probs[0]).argsort()

    msg = f'[{args.wav}]\n'
    for idx in sorted_indices[:args.topk]:
        msg += f'{ESC50.label_list[idx]}: {probs[0][idx]}\n'
    print(msg)
