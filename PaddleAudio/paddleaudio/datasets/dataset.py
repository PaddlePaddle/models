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


class ASRDataset(paddle.io.Dataset):
    """
    Base class of audio ASR dataset.
    """
    def __init__(self, feat_type: str, data: List[collections.namedtuple], **kwargs):
        super(ASRDataset, self).__init__()

        if feat_type not in feat_funcs.keys():
            raise RuntimeError(\
                f"Unknown feat_type: {feat_type}, it must be one in {list(feat_funcs.keys())}")

        self.feat_type = feat_type
        self.feat_config = kwargs  # Pass keyword arguments to customize feature config
        self.records = self._convert_to_records(data)

    def _convert_to_records(self, data: List[collections.namedtuple]):
        def _convert(sample):
            record = {}
            # To show all fields in a namedtuple: `type(sample)._fields`
            for field in type(sample)._fields:
                record[field] = getattr(sample, field)

            waveform, sr = load_audio(sample[0])  # The first element of sample is file path
            feat_func = feat_funcs[self.feat_type]
            feat = feat_func(waveform, sample_rate=sr, **self.feat_config) if feat_func else waveform
            record.update({'feat': feat, 'duration': len(waveform) / sr})
            return record

        records = list(tqdm(map(_convert, data), total=len(data)))
        return records

    def __getitem__(self, idx):
        record = self.records[idx]
        # TODO: To confirm what fields a ASR model wants.
        return np.array(record['feat']), np.array(record['text'])

    def __len__(self):
        return len(self.records)


class TTSDataset(paddle.io.Dataset):
    """
    Base class of audio TTS dataset.
    """
    def __init__(self, feat_type: str, data: List[collections.namedtuple], **kwargs):
        super(TTSDataset, self).__init__()
        raise NotImplementedError

    def _convert_to_records(self, data: List[collections.namedtuple]):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class DatasetFactory(object):
    """
    """
    task2cls = {
        'asr': ASRDataset,
        'tts': TTSDataset,
        'cls': AudioClassificationDataset,
    }

    def __init__(self, task: str, feat_type: 'str', data: List[collections.namedtuple], *args, **kwargs):

        if task is not None and task.lower() in self.task2cls:
            dataset_cls = self.task2cls[task.lower()]
        else:
            raise RuntimeError(f'Argement \'task\' must be one in {list(self.task2cls.keys())}, but got {task}')

        self._dataset = dataset_cls(feat_type, data, *args, **kwargs)  # Real dataset instance
        self.feat_config = self._dataset.feat_config

    @property
    def records(self):
        return self._dataset.records

    def __getitem__(self, idx):
        return self._dataset.__getitem__(idx)

    def __len__(self):
        return self._dataset.__len__()
