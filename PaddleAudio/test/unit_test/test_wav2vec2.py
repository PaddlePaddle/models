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

import numpy as np
import paddle
import paddleaudio
import pytest
from paddle.utils import download
from paddleaudio.models.wav2vec2 import Wav2Vec2ForCTC, Wav2Vec2Tokenizer

AUDIO_URL = 'https://bj.bcebos.com/paddleaudio/test/data/librispeech/sample1.flac'
TEXT_URL = 'https://bj.bcebos.com/paddleaudio/test/data/librispeech/sample1.txt'

EPS = 1e-4
# Test test-data are collected using audio dev_clean/1272/141231/1272-141231-0010.flac from librispeech
test_data = [
    ('wav2vec2-base-960h', -5.281228065490723, 8.575916290283203),
    ('wav2vec2-large-960h', -4.363298416137695, 10.152968406677246),
    ('wav2vec2-large-960h-lv60', -2.286926507949829, 4.644349575042725),
    ('wav2vec2-large-960h-lv60-self', -2.8010618686676025, 6.001106262207031),
]


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


def get_logic_from_model(name):
    x, _ = load_sample_audio_text()
    model = Wav2Vec2ForCTC(name, pretrained=True)
    model.eval()
    with paddle.no_grad():
        logits = model(x)
    return float(paddle.mean(logits)), float(paddle.std(logits))


@pytest.mark.parametrize('name,logit_mean,logit_std', test_data)
def test_acc(name, logit_mean, logit_std):

    mean, std = get_logic_from_model(name)
    assert abs(mean - logit_mean) < EPS
    assert abs(std - logit_std) < EPS


if __name__ == '__main__':
    test_acc(*test_data[0])
