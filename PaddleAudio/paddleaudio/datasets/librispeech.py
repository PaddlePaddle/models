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

import codecs
import collections
import json
import os
from typing import Dict, List, Tuple

from ..utils.download import decompress, download_and_decompress
from ..utils.env import DATA_HOME
from ..utils.log import logger
from .dataset import DatasetFactory

__all__ = ['LibriSpeech']


class LibriSpeech(DatasetFactory):
    """
    LibriSpeech
    """

    source_url = 'http://www.openslr.org/resources/12/'
    archieves = [
        {
            'url': source_url + 'train-clean-100.tar.gz',
            'md5': '2a93770f6d5c6c964bc36631d331a522',
        },
        {
            'url': source_url + 'train-clean-360.tar.gz',
            'md5': 'c0e676e450a7ff2f54aeade5171606fa',
        },
        {
            'url': source_url + 'train-other-500.tar.gz',
            'md5': 'd1a0fd59409feb2c614ce4d30c387708',
        },
        {
            'url': source_url + 'dev-clean.tar.gz',
            'md5': '42e2234ba48799c1f50f24a7926300a1',
        },
        {
            'url': source_url + 'dev-other.tar.gz',
            'md5': 'c8d0bcc9cca99d4f8b62fcc847357931',
        },
        {
            'url': source_url + 'test-clean.tar.gz',
            'md5': '32fa31d27d2e1cad72775fee3f4849a9',
        },
        {
            'url': source_url + 'test-other.tar.gz',
            'md5': 'fb5a50374b501bb3bac4815ee91d3135',
        },
    ]
    speaker_meta = os.path.join('LibriSpeech', 'SPEAKERS.TXT')
    utt_info = collections.namedtuple('META_INFO', ('file_path', 'utt_id', 'text', 'spk_id', 'spk_gender'))
    audio_path = 'LibriSpeech'
    manifest_path = os.path.join('LibriSpeech', 'manifest')
    subset = ['train-clean-100', 'train-clean-360', 'train-clean-500', \
            'dev-clean', 'dev-other', 'test-clean', 'test-other']

    def __init__(self, task: str = 'asr', subset: str = 'train-clean-100', feat_type: str = 'raw', **kwargs):
        """
        """
        assert subset in self.subset, 'Dataset subset must be one in train, dev and test, but got {}'.format(subset)
        self.subset = subset
        self.feat_type = feat_type
        super(LibriSpeech, self).__init__(task=task, feat_type=self.feat_type, data=self._get_data(), **kwargs)

    def _get_speaker_info(self) -> Dict[str, str]:
        ret = {}
        with open(os.path.join(DATA_HOME, self.speaker_meta), 'r') as rf:
            for line in rf.readlines():
                if ';' in line:  # Skip dataset abstract
                    continue
                spk_id, gender = map(str.strip, line.split('|')[:2])  # spk_id, gender
                ret.update({spk_id: gender})
        return ret

    def _get_text_info(self, trans_file) -> Dict[str, str]:
        ret = {}
        with open(trans_file, 'r') as rf:
            for line in rf.readlines():
                utt_id, text = map(str.strip, line.split(' ', 1))  # utt_id, text
                ret.update({utt_id: text})
        return ret

    def _get_data(self):
        if not os.path.isdir(os.path.join(DATA_HOME, self.audio_path)) or \
            not os.path.isfile(os.path.join(DATA_HOME, self.speaker_meta)):
            download_and_decompress(self.archieves, DATA_HOME, len(self.archieves))

        # Speaker info
        speaker_info = self._get_speaker_info()

        # Text info
        text_info = {}
        for root, _, files in os.walk(os.path.join(DATA_HOME, self.audio_path, self.subset)):
            for file in files:
                if file.endswith('.trans.txt'):
                    text_info.update(self._get_text_info(os.path.join(root, file)))

        data = []
        for root, _, files in os.walk(os.path.join(DATA_HOME, self.audio_path, self.subset)):
            for file in files:
                if file.endswith('.flac'):
                    utt_id = os.path.splitext(file)[0]
                    spk_id = utt_id.split('-')[0]
                    if utt_id not in text_info \
                        or spk_id not in speaker_info :  # Skip samples with incomplete data
                        continue
                    file_path = os.path.join(root, file)
                    text = text_info[utt_id]
                    spk_gender = speaker_info[spk_id]
                    data.append(self.utt_info(file_path, utt_id, text, spk_id, spk_gender))

        return data

    def create_manifest(self, prefix='manifest'):
        if not os.path.isdir(os.path.join(DATA_HOME, self.manifest_path)):
            os.makedirs(os.path.join(DATA_HOME, self.manifest_path))

        manifest_file = os.path.join(DATA_HOME, self.manifest_path, f'{prefix}.{self.subset}')
        with codecs.open(manifest_file, 'w', 'utf-8') as f:
            for record in self.records:
                record_line = json.dumps(
                    {
                        'utt': record['utt_id'],
                        'feat': record['file_path'],
                        'feat_shape': (record['duration'], ),
                        'text': record['text'],
                        'spk': record['spk_id'],
                        'gender': record['spk_gender'],
                    },
                    ensure_ascii=False)
                f.write(record_line + '\n')
        logger.info(f'Manifest file {manifest_file} created.')
