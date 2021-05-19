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

__all__ = ['Aishell']


class Aishell(DatasetFactory):
    """
    Aishell
    """

    archieves = [
        {
            'url': 'http://www.openslr.org/resources/33/data_aishell.tgz',
            'md5': '2f494334227864a8a8fec932999db9d8',
        },
    ]
    text_meta = os.path.join('data_aishell', 'transcript', 'aishell_transcript_v0.8.txt')
    utt_info = collections.namedtuple('META_INFO', ('file_path', 'utt_id', 'text'))
    audio_path = os.path.join('data_aishell', 'wav')
    manifest_path = os.path.join('data_aishell', 'manifest')
    subset = ['train', 'dev', 'test']

    def __init__(self, task: str = 'asr', subset: str = 'train', feat_type: str = 'raw', **kwargs):
        """
        """
        assert subset in self.subset, 'Dataset subset must be one in train, dev and test, but got {}'.format(subset)
        self.subset = subset
        self.feat_type = feat_type
        super(Aishell, self).__init__(task=task, feat_type=self.feat_type, data=self._get_data(), **kwargs)

    def _get_text_info(self) -> Dict[str, str]:
        ret = {}
        with open(os.path.join(DATA_HOME, self.text_meta), 'r') as rf:
            for line in rf.readlines()[1:]:
                utt_id, text = map(str.strip, line.split(' ', 1))  # utt_id, text
                ret.update({utt_id: ''.join(text.split())})
        return ret

    def _get_data(self):
        if not os.path.isdir(os.path.join(DATA_HOME, self.audio_path)) or \
            not os.path.isfile(os.path.join(DATA_HOME, self.text_meta)):
            download_and_decompress(self.archieves, DATA_HOME)
            # Extract *wav from *.tar.gz.
            for root, _, files in os.walk(os.path.join(DATA_HOME, self.audio_path)):
                for file in files:
                    if file.endswith('.tar.gz'):
                        decompress(os.path.join(root, file))
                        os.remove(os.path.join(root, file))

        text_info = self._get_text_info()

        data = []
        for root, _, files in os.walk(os.path.join(DATA_HOME, self.audio_path, self.subset)):
            for file in files:
                if file.endswith('.wav'):
                    utt_id = os.path.splitext(file)[0]
                    if utt_id not in text_info:  # There are some utt_id that without label
                        continue
                    text = text_info[utt_id]
                    file_path = os.path.join(root, file)
                    data.append(self.utt_info(file_path, utt_id, text))

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
                        'text': record['text']
                    },
                    ensure_ascii=False)
                f.write(record_line + '\n')
        logger.info(f'Manifest file {manifest_file} created.')
