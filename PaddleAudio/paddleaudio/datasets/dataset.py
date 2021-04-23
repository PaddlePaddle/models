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

import os
from typing import List, Tuple

import librosa
import numpy as np
import paddle
from tqdm import tqdm

from ..features import linear_spect, log_spect, mel_spect
from ..utils.log import logger


class AudioClassificationDataset(paddle.io.Dataset):
    """
    Base class of audio classification dataset.
    """
    _feat_func = {
        'raw': None,
        'mel_spect': mel_spect,
        'linear_spect': linear_spect,
        'log_spect': log_spect,
    }

    def __init__(self, files: List[str], labels: List[int], sample_rate: int, feat_type: str = 'raw', **kwargs):
        """
        Ags:
            files (:obj:`List[str]`): A list of absolute path of audio files.
            labels (:obj:`List[int]`): Labels of audio files.
            sample_rate (:obj:`int`): Sample rate of audio files.
            feat_type (:obj:`str`, `optional`, defaults to `raw`):
                It identifies the feature type that user wants to extrace of an audio file.
        """
        super(AudioClassificationDataset, self).__init__()

        if feat_type not in self._feat_func.keys():
            raise RuntimeError(\
                f"Unknown feat_type: {feat_type}, it must be one in {list(self._feat_func.keys())}")
        self.feat_type = feat_type

        self.files = files
        self.labels = labels
        self.records = self._convert_to_records(sample_rate, **kwargs)

    def _get_data(self, input_file: str):
        raise NotImplementedError

    def _convert_to_records(self, sample_rate: int, **kwargs) -> List[dict]:
        records = []
        feat_func = self._feat_func[self.feat_type]

        logger.info('Start extracting features from audio files.')
        for file, label in tqdm(zip(self.files, self.labels), total=len(self.files)):
            record = {}
            waveform, _ = librosa.load(file, sr=sample_rate)
            record['feat'] = feat_func(waveform, **kwargs) if feat_func else waveform
            record['label'] = label
            records.append(record)

        return records

    def __getitem__(self, idx):
        record = self.records[idx]
        return np.array(record['feat']), np.array(record['label'], dtype=np.int64)

    def __len__(self):
        return len(self.records)
