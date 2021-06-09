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
import glob
import tempfile

import numpy as np
import paddle
import paddleaudio
from paddle.utils import download
from paddleaudio.backends.audio import load, normalize
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


#from paddleaudio.datasets import LIBRISPEECH
#LIBRISPEECH('test-clean')#


def get_text(line):
    t = ' '.join(line.split()[1:])
    t = t.lower()
    return t


def get_name_txt(trans_files):
    name2txt = {}
    for file in trans_files:
        with open(file) as f:
            lines = f.read().split('\n')
            d = {l.split()[0]: get_text(l) for l in lines if len(l) > 0}
            name2txt.update(d)
    return name2txt


def levenshtein(s, t):
    """
    Compute levenshtein distance between two sequence
    Reference:
        https://www.python-course.eu/levenshtein_distance.php
    """

    rows = len(s) + 1
    cols = len(t) + 1
    dist = [[0 for x in range(cols)] for x in range(rows)]

    # source prefixes can be transformed into empty strings
    # by deletions:
    for i in range(1, rows):
        dist[i][0] = i

    # target prefixes can be created from an empty source string
    # by inserting the characters
    for i in range(1, cols):
        dist[0][i] = i

    for col in range(1, cols):
        for row in range(1, rows):
            if s[row - 1] == t[col - 1]:
                cost = 0
            else:
                cost = 1
            dist[row][col] = min(
                dist[row - 1][col] + 1,  # deletion
                dist[row][col - 1] + 1,  # insertion
                dist[row - 1][col - 1] + cost)  # substitution

    return dist[row][col]


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

    parser.add_argument(
        '-t', '--test_path', type=str, required=False, default='./test_clean')
    parser.add_argument('-o', '--output', type=str, required=False, default='')
    parser.add_argument(
        '-l', '--log_interval', type=int, required=False, default=20)

    args = parser.parse_args()

    wav_files = glob.glob(f'{args.test_path}/*/*/*.flac')
    trans_files = glob.glob(f'{args.test_path}/*/*/*.txt')

    if len(wav_files) == 0:
        logger.error(f'Not wav file found in  {args.test_path}')
        exit

    if len(trans_files) == 0:
        logger.error(f'Not transcrition file found in  {args.test_path}')
        exit

    logger.info(f'{len(wav_files)} wav files found in  {args.test_path}')
    logger.info(
        f'{len(trans_files)} transcrition files found in  {args.test_path}')

    name2txt = get_name_txt(trans_files)
    for f in wav_files:
        key = f.split('/')[-1][:-5]
        if not name2txt.get(key, None):
            raise Exception(
                f'{key} is presented in transcrition but the wav file is not found, \
                make sure the data is completely downloaded')

    logger.info(f'Using device {args.device}')
    paddle.set_device(args.device)

    logger.info(f'Loading model {args.config}...')

    model = Wav2Vec2ForCTC(args.config, pretrained=True)
    tokenizer = Wav2Vec2Tokenizer()

    if args.output == '':
        output_file = tempfile.NamedTemporaryFile('wt', suffix='txt').name
        output_file = output_file
    else:
        output_file = args.output
    logger.info(f'Results will be saved to {output_file}')

    dst_f = open(output_file, 'wt')
    msg = f'model: {args.config}\n'
    msg += f'data: {args.test_path}\n'
    dst_f.write(msg)

    avg_cer = 0
    avg_wer = 0

    with paddle.no_grad():
        for i, f in enumerate(wav_files):
            x = load_audio(f)
            logits = model(x)
            # get the token index prediction
            idx = paddle.argmax(logits, -1)
            # decode to text
            pred = tokenizer.decode(idx[0])

            key = f.split('/')[-1][:-5]
            true = name2txt[key]
            dst_f.write(f'{key}|{pred}|{true}\n')

            cer = levenshtein(pred, true) / len(true)
            wer = levenshtein(pred.split(), true.split()) / len(true.split())
            avg_cer = (avg_cer * i + cer) / (i + 1)
            avg_wer = (avg_wer * i + wer) / (i + 1)
            if i % args.log_interval == 0:
                logger.info(f'{i}|{len(wav_files)},pred==> {pred}')
                logger.info(f'{i}|{len(wav_files)},true==> {true}')
                logger.info(f'avg_cer = {avg_cer}, avg_wer = {avg_wer}')

    msg = f'final avg_cer = {avg_cer}, final avg_wer = {avg_wer}'
    dst_f.write(msg + '\n')
    dst_f.close()

    logger.info(msg)
    logger.info(f'Results have been saved to {output_file}')
