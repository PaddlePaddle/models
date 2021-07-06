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
import csv
import glob
import os
import random
from typing import Dict, List, Tuple

from paddle.io import Dataset
from tqdm import tqdm

from ..backends import load as load_audio
from ..utils.download import decompress, download_and_decompress
from ..utils.env import DATA_HOME
from ..utils.log import logger
from .dataset import feat_funcs

__all__ = ['VoxCeleb1']


class VoxCeleb1(Dataset):
    source_url = 'https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/'
    archieves_audio_dev = [
        {
            'url': source_url + 'vox1_dev_wav_partaa',
            'md5': 'e395d020928bc15670b570a21695ed96',
        },
        {
            'url': source_url + 'vox1_dev_wav_partab',
            'md5': 'bbfaaccefab65d82b21903e81a8a8020',
        },
        {
            'url': source_url + 'vox1_dev_wav_partac',
            'md5': '017d579a2a96a077f40042ec33e51512',
        },
        {
            'url': source_url + 'vox1_dev_wav_partad',
            'md5': '7bb1e9f70fddc7a678fa998ea8b3ba19',
        },
    ]
    archieves_audio_test = [
        {
            'url': source_url + 'vox1_test_wav.zip',
            'md5': '185fdc63c3c739954633d50379a3d102',
        },
    ]
    archieves_meta = [
        {
            'url':
            'https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/iden_split.txt',
            'md5': '09710b779bc221585f5837ef0341a1d2',
        },
    ]

    num_speakers = 1251  # For speaker identification task
    meta_info = collections.namedtuple(
        'META_INFO', ('id', 'duration', 'wav', 'start', 'stop', 'spk_id'))
    base_path = os.path.join(DATA_HOME, 'vox1')
    wav_path = os.path.join(base_path, 'wav')
    meta_path = os.path.join(base_path, 'meta')
    iden_split_file = os.path.join(meta_path, 'iden_split.txt')
    csv_path = os.path.join(base_path, 'csv')
    subsets = ['train', 'dev', 'test']

    def __init__(self,
                 subset: str = 'train',
                 feat_type: str = 'raw',
                 random_chunk: bool = True,
                 chunk_duration: float = 3.0,
                 seed: int = 0,
                 **kwargs):

        assert subset in self.subsets, \
            'Dataset subset must be one in {}, but got {}'.format(self.subsets, subset)

        self.subset = subset
        self.spk_id2label = {}
        self.feat_type = feat_type
        self.feat_config = kwargs
        self._data = self._get_data()
        self.random_chunk = random_chunk
        self.chunk_duration = chunk_duration
        super(VoxCeleb1, self).__init__()

        # Set up a seed to reproduce training or predicting result.
        random.seed(seed)

    def _get_data(self):
        # Download audio files.
        if not os.path.isdir(self.base_path):
            download_and_decompress(
                self.archieves_audio_dev, self.base_path, decompress=False)

            download_and_decompress(
                self.archieves_audio_test, self.base_path, decompress=True)

            # Download all parts and concatenate the files into one zip file.
            # The result is same as using the command `cat vox1_dev* > vox1_dev_wav.zip`.
            dev_zipfile = os.path.join(self.base_path, 'vox1_dev_wav.zip')
            with open(dev_zipfile, 'wb') as f:
                for part_of_zip in glob.glob(
                        os.path.join(self.base_path, 'vox1_dev_wav_parta*')):
                    with open(part_of_zip, 'rb') as p:
                        f.write(p.read())

            # Extract all audio files of dev and test set.
            decompress(dev_zipfile, self.base_path)

        # Download meta files.
        if not os.path.isdir(self.meta_path):
            download_and_decompress(
                self.archieves_meta, self.meta_path, decompress=False)

        # Data preparation.
        if not os.path.isdir(self.csv_path):
            os.makedirs(self.csv_path)
            self.prepare_data()

        spk_id_set = set()
        data = []
        with open(os.path.join(self.csv_path, f'{self.subset}.csv'), 'r') as rf:
            for line in rf.readlines()[1:]:
                audio_id, duration, wav, start, stop, spk_id = line.strip(
                ).split(',')
                spk_id_set.add(spk_id)
                data.append(
                    self.meta_info(audio_id, float(duration), wav, int(start),
                                   int(stop), spk_id))
        for idx, uniq_spk_id in enumerate(sorted(list(spk_id_set))):
            self.spk_id2label[uniq_spk_id] = idx

        return data

    def _convert_to_record(self, idx: int):
        sample = self._data[idx]

        record = {}
        # To show all fields in a namedtuple: `type(sample)._fields`
        for field in type(sample)._fields:
            record[field] = getattr(sample, field)

        waveform, sr = load_audio(
            record['wav'])  # The first element of sample is file path

        if self.subset == 'train' and self.random_chunk:
            num_wav_samples = waveform.shape[0]
            num_chunk_samples = int(self.chunk_duration * sr)
            start = random.randint(0, num_wav_samples - num_chunk_samples - 1)
            stop = start + num_chunk_samples
        else:
            start = record['start']
            stop = record['stop']

        waveform = waveform[start:stop]

        assert self.feat_type in feat_funcs.keys(), \
            f"Unknown feat_type: {self.feat_type}, it must be one in {list(feat_funcs.keys())}"
        feat_func = feat_funcs[self.feat_type]
        feat = feat_func(
            waveform, sr=sr, **self.feat_config) if feat_func else waveform
        record.update({
            'feat': feat,
            'label': self.spk_id2label[record['spk_id']]
        })
        return record

    @staticmethod
    def _get_chunks(seg_dur, audio_id, audio_duration):
        num_chunks = int(audio_duration / seg_dur)  # all in milliseconds

        chunk_lst = [
            audio_id + "_" + str(i * seg_dur) + "_" + str(i * seg_dur + seg_dur)
            for i in range(num_chunks)
        ]
        return chunk_lst

    def _get_audio_info(self, wav_file: str) -> List[List[str]]:
        waveform, sr = load_audio(wav_file)
        spk_id, sess_id, utt_id = wav_file.split("/")[-3:]
        audio_id = '-'.join([spk_id, sess_id, utt_id.split(".")[0]])
        audio_duration = waveform.shape[0] / sr

        # Split into pieces of self.chunk_duration seconds.
        ret = []
        uniq_chunks_list = self._get_chunks(self.chunk_duration, audio_id,
                                            audio_duration)
        for chunk in uniq_chunks_list:
            s, e = chunk.split("_")[-2:]  # Timestamps of start and end
            start_sample = int(float(s) * sr)
            end_sample = int(float(e) * sr)
            # id, duration, wav, start, stop, spk_id
            ret.append([
                chunk, audio_duration, wav_file, start_sample, end_sample,
                spk_id
            ])
        return ret

    def generate_csv(self, wav_files: List[str], output_file: str):
        logger.info(f'Generating csv: {output_file}')
        header = ["id", "duration", "wav", "start", "stop", "spk_id"]

        infos = list(
            tqdm(map(self._get_audio_info, wav_files), total=len(wav_files)))

        csv_lines = []
        for info in infos:
            csv_lines.extend(info)

        with open(output_file, mode="w") as csv_f:
            csv_writer = csv.writer(
                csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(header)
            for line in csv_lines:
                csv_writer.writerow(line)

    def prepare_data(self):
        train_files, dev_files, test_files = [], [], []
        id2subset = {1: train_files, 2: dev_files, 3: test_files}
        with open(self.iden_split_file, 'r') as f:
            for line in f.readlines():
                subset_id, rel_file = line.strip().split(' ')
                abs_file = os.path.join(self.wav_path, rel_file)
                id2subset[int(subset_id)].append(abs_file)

        self.generate_csv(train_files, os.path.join(self.csv_path, 'train.csv'))
        self.generate_csv(dev_files, os.path.join(self.csv_path, 'dev.csv'))
        self.generate_csv(test_files, os.path.join(self.csv_path, 'test.csv'))

    def __getitem__(self, idx):
        return self._convert_to_record(idx)

    def __len__(self):
        return len(self._data)
