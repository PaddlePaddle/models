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

import argparse
import glob
import multiprocessing
import os
from os import PathLike
from typing import List, Union

import h5py
import librosa
import numpy as np
import paddleaudio as pa
import yaml
from paddleaudio.utils.log import Logger

logger = Logger(__file__)


class FeatureExtractor:
    def __init__(self, **kwargs):

        self.transform = pa.transforms.MelSpectrogram(**kwargs)

    def process_wav(self, wav: Union[PathLike, np.ndarray]) -> np.ndarray:

        if isinstance(wav, str):
            wav, sr = librosa.load(wav, sr=None)
            #wav, sr = pa.load(wav, sr=None)
            target_sr = self.kwargs.get('sr')
            assert sr == target_sr, f'sr: {sr} ~= {target_sr}'

        if wav.dtype == 'int16':
            wav = pa.depth_convert(wav, 'float32')
        wav = paddle.to_tensor(wav).unsqueeze(0)
        x = self.transform(wav)
        return x


def wav_list_to_fbank(wav_list: List[PathLike],
                      key_list: List[str],
                      dst_file: PathLike,
                      feature_extractor: FeatureExtractor) -> None:
    """Convert wave list to fbank, store into an h5 file
    """

    logger.info(f'saving to {dst_file}')
    dst_h5_obj = h5py.File(dst_file, "w")
    logger.info(f'{len(wav_list)} wav files listed')
    for f, key in zip(wav_list, key_list):
        x = feature_extractor.process_wav(f)
        dst_h5_obj.create_dataset(key, data=x)
    dst_h5_obj.close()


def wav_list_to_fbank_mp(params):
    """Convert wave list to fbank, store into an h5 file, multiprocessing warping"""

    wav_list, key_list, dst_file, feature_extractor = params
    wav_list_to_fbank(wav_list, key_list, dst_file, feature_extractor)


def read_scp(scp_file):
    with open(scp_file) as f:
        lines = f.read().split('\n')

# import pdb;pdb.set_trace()
    names = [l.split()[0] for l in lines if len(l) > 1]
    files = [l.split()[1] for l in lines if len(l) > 1]
    return names, files

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='wave2mel')
    parser.add_argument(
        '-c', '--config', type=str, required=True, default='config.yaml')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    paddle.set_device(config['device'])
    feature_extractor = FeatureExtractor(**config['fbank'])

    names, wav_files = read_scp(config['wav_scp'])
    n_file_per_h5 = config['h5']['n_wavs']
    if n_file_per_h5 == -1:
        logger.info('no grouping')
        n_group = 1
        group_files = [wav_files]
        group_names = [names]

    else:
        logger.info(f'grouping {n_file_per_h5} files into one h5 file')
        n_group = len(wav_files) // n_file_per_h5 + 1
        group_files = [
            wav_files[i * n_file_per_h5:(i + 1) * n_file_per_h5]
            for i in range(n_group)
        ]
        group_names = [
            names[i * n_file_per_h5:(i + 1) * n_file_per_h5]
            for i in range(n_group)
        ]

    os.makedirs(config['h5']['output_folder'], exist_ok=True)
    if config['num_works'] <= 1:
        for i in range(n_group):
            logger.info(f'processing group {i}/{n_group}')
            prefix = config['h5']['prefix']
            dst_file = os.path.join(config['h5']['output_folder'],
                                    f'{prefix}-{i:05}.h5')
            logger.info(f'saving file to {dst_file}')
            wav_list_to_fbank(group_files[i], group_names[i], dst_file,
                              feature_extractor)
    else:
        pool = multiprocessing.Pool(config['num_works'])
        dst_files = []
        # Collect multi-processing parameters
        for i in range(n_group):
            prefix = config['h5']['prefix']
            dst_file = os.path.join(config['h5']['output_folder'],
                                    f'{prefix}-{i:05}.h5')
            dst_files.append(dst_file)
            params = [(file, name, dst_file, feature_extractor)
                      for file, name, dst_file in zip(group_files, group_names,
                                                      dst_files)]

        pool.map(wav_list_to_fbank_mp, params)
        pool.close()
        pool.join()
