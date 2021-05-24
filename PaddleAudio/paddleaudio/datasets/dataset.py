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

import collections
import os
from typing import Dict, List, Tuple

import numpy as np
import paddle
from pathos.multiprocessing import ProcessPool
from pathos.threading import ThreadPool
from tqdm import tqdm

from ..backends import load as load_audio
from ..features import linear_spect, log_spect, mel_spect

feat_funcs = {
    'raw': None,
    'mel_spect': mel_spect,
    'linear_spect': linear_spect,
    'log_spect': log_spect,
}


class AudioClassificationDataset(paddle.io.Dataset):
    """
    Base class of audio classification dataset.
    """
    def __init__(self,
                 files: List[str],
                 labels: List[int],
                 sample_rate: int,
                 duration: float,
                 feat_type: str = 'raw',
                 **kwargs):
        """
        Ags:
            files (:obj:`List[str]`): A list of absolute path of audio files.
            labels (:obj:`List[int]`): Labels of audio files.
            sample_rate (:obj:`int`): Sample rate of audio files.
            duration (:obj:`float`): Duration of audio files.
            feat_type (:obj:`str`, `optional`, defaults to `raw`):
                It identifies the feature type that user wants to extrace of an audio file.
        """
        super(AudioClassificationDataset, self).__init__()

        if feat_type not in feat_funcs.keys():
            raise RuntimeError(\
                f"Unknown feat_type: {feat_type}, it must be one in {list(feat_funcs.keys())}")

        self.files = files
        self.labels = labels
        self.sample_rate = sample_rate
        self.duration = duration

        self.feat_type = feat_type
        self.feat_config = kwargs  # Pass keyword arguments to customize feature config

    def _get_data(self, input_file: str):
        raise NotImplementedError

    def _convert_to_record(self, idx):
        file, label = self.files[idx], self.labels[idx]

        waveform, _ = load_audio(file, sr=self.sample_rate)
        normal_length = self.sample_rate * self.duration
        if len(waveform) > normal_length:
            waveform = waveform[:normal_length]
        else:
            waveform = np.pad(waveform, (0, normal_length - len(waveform)))

        feat_func = feat_funcs[self.feat_type]

        record = {}
        record['feat'] = feat_func(waveform, sample_rate=self.sample_rate, **
                                   self.feat_config) if feat_func else waveform
        record['label'] = label
        return record

    def __getitem__(self, idx):
        record = self._convert_to_record(idx)
        return np.array(record['feat']).transpose(), np.array(record['label'], dtype=np.int64)

    def __len__(self):
        return len(self.files)
