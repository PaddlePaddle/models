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
from typing import List, Tuple

from ..utils.download import download_and_decompress
from ..utils.env import DATA_HOME
from ..utils.log import logger
from .dataset import AudioClassificationDataset

__all__ = ['ESC50']


class ESC50(AudioClassificationDataset):
    """
    Environment Sound Classification Dataset
    """

    archieves = [
        {
            'url': 'https://github.com/karoldvl/ESC-50/archive/master.zip',
            'md5': '70aba3bada37d2674b8f6cd5afd5f065',
        },
    ]
    meta = os.path.join('ESC-50-master', 'meta', 'esc50.csv')
    meta_info = collections.namedtuple('META_INFO',
                                       ('filename', 'fold', 'target', 'category', 'esc10', 'src_file', 'take'))
    audio_path = os.path.join('ESC-50-master', 'audio')
    sample_rate = 44100  # 44.1 khz
    duration = 5  # 5s

    def __init__(self, mode: str = 'train', split: int = 1, feat_type: str = 'raw', **kwargs):
        """
        Ags:
            mode (:obj:`str`, `optional`, defaults to `train`):
                It identifies the dataset mode (train or dev).
            split (:obj:`int`, `optional`, defaults to 1):
                It specify the fold of dev dataset.
            feat_type (:obj:`str`, `optional`, defaults to `raw`):
                It identifies the feature type that user wants to extrace of an audio file.
        """
        files, labels = self._get_data(mode, split)
        super(ESC50, self).__init__(files=files,
                                    labels=labels,
                                    sample_rate=self.sample_rate,
                                    feat_type=feat_type,
                                    **kwargs)

    def _get_meta_info(self) -> List[collections.namedtuple]:
        ret = []
        with open(os.path.join(DATA_HOME, self.meta), 'r') as rf:
            for line in rf.readlines()[1:]:
                ret.append(self.meta_info(*line.strip().split(',')))
        return ret

    def _get_data(self, mode: str, split: int) -> Tuple[List[str], List[int]]:
        if not os.path.isdir(os.path.join(DATA_HOME, self.audio_path)) or \
            not os.path.isfile(os.path.join(DATA_HOME, self.meta)):
            download_and_decompress(self.archieves, DATA_HOME)

        meta_info = self._get_meta_info()

        files = []
        labels = []
        for sample in meta_info:
            filename, fold, target, _, _, _, _ = sample
            if mode == 'train' and int(fold) != split:
                files.append(os.path.join(DATA_HOME, self.audio_path, filename))
                labels.append(int(target))

            if mode != 'train' and int(fold) == split:
                files.append(os.path.join(DATA_HOME, self.audio_path, filename))
                labels.append(int(target))

        return files, labels
