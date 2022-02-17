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

import numpy as np
import paddle
import paddleaudio
from paddle.utils import download
from paddleaudio.backends.audio import normalize
from paddleaudio.models.wav2vec2 import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
from paddleaudio.utils.log import Logger

logger = Logger(__file__)

AUDIO_URL = 'https://bj.bcebos.com/paddleaudio/test/data/librispeech/sample1.flac'
TEXT_URL = 'https://bj.bcebos.com/paddleaudio/test/data/librispeech/sample1.txt'


def load_audio(file):
    """Load audio from local path
    The function will resample the audio to 16K and re-normalize it to have zero-mean and unit-variance
    """
    s, _ = paddleaudio.load(file, sr=16000, normal=True, norm_type='gaussian')
    x = paddle.to_tensor(s)
    x = x.unsqueeze(0)
    return x


def load_sample_audio_text():
    """Load sample audio and text"""
    text_path = download.get_weights_path_from_url(TEXT_URL)
    with open(text_path) as f:
        text = f.read()

    audio_path = download.get_weights_path_from_url(AUDIO_URL)
    x = load_audio(audio_path)
    return x, text


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='testing wav2vec2.0')
    parser.add_argument(
        '-d',
        '--device',
        choices=['cpu', 'gpu'],
        default="gpu",
        help="Select which device to train model, defaults to gpu.")
    parser.add_argument(
        '-c',
        '--config',
        type=str,
        required=False,
        default='wav2vec2-base-960h')
    parser.add_argument('-a', '--audio', type=str, required=False, default='')
    args = parser.parse_args()

    logger.info(f'Using device {args.device}')
    paddle.set_device(args.device)

    model = Wav2Vec2ForCTC(args.config, pretrained=True)
    tokenizer = Wav2Vec2Tokenizer()

    if args.audio == '':
        x, text = load_sample_audio_text()
    else:
        x = load_audio(args.audio)
        text = None

    with paddle.no_grad():
        logits = model(x)
    # get the token index prediction
    idx = paddle.argmax(logits, -1)
    # decode to text
    pred = tokenizer.decode(idx[0])

    logger.info(f'pred==> {pred}')
    logger.info(f'true==> {text}')
