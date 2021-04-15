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
from typing import List

import librosa
import numpy as np
import paddle

from paddleaudio.features import mel_spect
from paddleaudio.models import CNN14
from paddleaudio.utils.log import logger

parser = argparse.ArgumentParser(__doc__)
# features
parser.add_argument("--sr", type=int, default=32000, help="Sample rate of inference audio.")
parser.add_argument('--window_size', type=int, default=1024)
parser.add_argument('--hop_size', type=int, default=320)
parser.add_argument('--mel_bins', type=int, default=64)
parser.add_argument('--fmin', type=int, default=50)
parser.add_argument('--fmax', type=int, default=14000)
# waveform
parser.add_argument("--wav", type=str, required=True, help="Audio file to infer.")
parser.add_argument('--sample_duration', type=float, default=1.0)  # 1s
parser.add_argument('--hop_duration', type=float, default=0.3)  # 0.3s

parser.add_argument("--output_dir", type=str, default='./output_dir')
parser.add_argument("--use_gpu",
                    type=ast.literal_eval,
                    default=True,
                    help="Whether use GPU for fine-tuning, input should be True or False")
parser.add_argument("--checkpoint", type=str, default='./assets/cnn14.pdparams', help="Checkpoint of model.")
args = parser.parse_args()


def split(waveform: np.ndarray, win_size: int, hop_size: int):
    """
    Split into N audios.
    N is decided by win_size and hop_size.
    """
    assert isinstance(waveform, np.ndarray)
    ret = []
    for i in range(0, len(waveform), hop_size):
        segment = waveform[i:i + win_size]
        if len(segment) < win_size:
            segment = np.pad(segment, (0, win_size - len(segment)))
        ret.append(segment)
    return ret


def batchify(data: List[List[float]], batch_size: int):
    """
    Extract features from waveforms and create batches.
    """
    examples = []
    for waveform in data:
        feat = mel_spect(
            waveform,
            sample_rate=args.sr,
            window_size=args.window_size,
            hop_length=args.hop_size,
            mel_bins=args.mel_bins,
            fmin=args.fmin,
            fmax=args.fmax,
        )
        examples.append(np.expand_dims(feat.transpose(), 0))  # (mel_bins, time) -> (1, time, mel_bins)

    # Seperates data into some batches.
    one_batch = []
    for example in examples:
        one_batch.append(example)
        if len(one_batch) == batch_size:
            yield one_batch
            one_batch = []
    if one_batch:
        yield one_batch


def predict(model, data: List[List[float]], batch_size: int = 1, use_gpu: bool = False):

    paddle.set_device('gpu') if use_gpu else paddle.set_device('cpu')

    batches = batchify(data, batch_size)
    results = None
    model.eval()
    for batch in batches:
        feats = paddle.to_tensor(batch)
        audioset_scores = model(feats)
        if results is None:
            results = audioset_scores.numpy()
        else:
            results = np.concatenate((results, audioset_scores.numpy()))

    return results


if __name__ == '__main__':
    model = CNN14(extract_embedding=False, checkpoint=args.checkpoint)
    waveform = librosa.load(args.wav, sr=args.sr)[0]
    data = split(waveform, int(args.sample_duration * args.sr), int(args.hop_duration * args.sr))
    results = predict(model, data, batch_size=8, use_gpu=args.use_gpu)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    time = np.arange(0, 1, int(args.hop_duration * args.sr) / len(waveform))
    output_file = os.path.join(args.output_dir, f'audioset_tagging_sr_{args.sr}.npz')
    np.savez(output_file, time=time, scores=results)
    logger.info(f'Saved tagging results to {output_file}')
