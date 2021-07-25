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

import glob
import json
import os
import subprocess
import time
import warnings

import h5py
import librosa
import numpy as np
import paddle
import paddleaudio
import yaml
from paddle.io import DataLoader, Dataset, IterableDataset
from paddleaudio.utils import augments
from utils import get_labels, get_ytid_clsidx_mapping


def spect_permute(spect, tempo_axis, nblocks):
    """spectrogram  permutaion"""
    assert spect.ndim == 2., 'only supports 2d tensor or numpy array'
    if tempo_axis == 0:
        nt, nf = spect.shape
    else:
        nf, nt = spect.shape
    if nblocks <= 1:
        return spect

    block_width = nt // nblocks + 1
    if tempo_axis == 1:
        blocks = [
            spect[:, block_width * i:(i + 1) * block_width]
            for i in range(nblocks)
        ]
        np.random.shuffle(blocks)
        new_spect = np.concatenate(blocks, 1)
    else:
        blocks = [
            spect[block_width * i:(i + 1) * block_width, :]
            for i in range(nblocks)
        ]
        np.random.shuffle(blocks)
        new_spect = np.concatenate(blocks, 0)
    return new_spect


def random_choice(a):
    i = np.random.randint(0, high=len(a))
    return a[int(i)]


def get_keys(file_pointers):
    all_keys = []
    key2file = {}
    for fp in file_pointers:
        all_keys += list(fp.keys())
        key2file.update({k: fp for k in fp.keys()})
    return all_keys, key2file


class H5AudioSet(Dataset):
    """
    Dataset class for Audioset, with mel features stored in multiple hdf5 files.
    The h5 files store mel-spectrogram features pre-extracted from wav files.
    Use wav2mel.py to do feature extraction.
    """
    def __init__(self,
                 h5_files,
                 config,
                 augment=True,
                 training=True,
                 balanced_sampling=True):
        super(H5AudioSet, self).__init__()
        self.h5_files = h5_files
        self.config = config
        self.file_pointers = [h5py.File(f) for f in h5_files]
        self.all_keys, self.key2file = get_keys(self.file_pointers)
        self.augment = augment
        self.training = training
        self.balanced_sampling = balanced_sampling
        print(
            f'{len(self.h5_files)} h5 files, totally {len(self.all_keys)} audio files listed'
        )
        self.ytid2clsidx, self.clsidx2ytid = get_ytid_clsidx_mapping()

    def _process(self, x):
        assert x.shape[0] == self.config[
            'mel_bins'], 'the first dimension must be mel frequency'

        target_len = self.config['max_mel_len']
        if x.shape[1] <= target_len:
            pad_width = (target_len - x.shape[1]) // 2 + 1
            x = np.pad(x, ((0, 0), (pad_width, pad_width)))
        x = x[:, :target_len]

        if self.training and self.augment:
            x = augments.random_crop2d(x,
                                       self.config['mel_crop_len'],
                                       tempo_axis=1)
            x = spect_permute(x, tempo_axis=1, nblocks=random_choice([0, 2, 3]))
            aug_level = random_choice([0.2, 0.1, 0])
            x = augments.adaptive_spect_augment(x,
                                                tempo_axis=1,
                                                level=aug_level)
        return x.T

    def __getitem__(self, idx):

        if self.balanced_sampling:
            cls_id = int(np.random.randint(0, self.config['num_classes']))
            keys = self.clsidx2ytid[cls_id]
            k = random_choice(self.all_keys)
            cls_ids = self.ytid2clsidx[k]
        else:
            idx = idx % len(self.all_keys)
            k = self.all_keys[idx]
            cls_ids = self.ytid2clsidx[k]
        fp = self.key2file[k]
        x = fp[k][:, :]
        x = self._process(x)

        y = np.zeros((self.config['num_classes'], ), 'float32')
        y[cls_ids] = 1.0

        return x, y

    def __len__(self):
        return len(self.all_keys)


def get_ytid2labels(segment_csv):
    """
    compute the mapping (dict object) from youtube id to audioset labels.
    """
    with open(segment_csv) as F:
        lines = F.read().split('\n')

    lines = [l for l in lines if len(l) > 0 and l[0] != '#']
    ytid2labels = {l.split(',')[0]: l.split('"')[-2] for l in lines}
    return ytid2labels


def worker_init(worker_id):

    time.sleep(worker_id / 32)
    np.random.seed(int(time.time()) % 100 + worker_id)


def get_train_loader(config):

    train_h5_files = glob.glob(config['unbalanced_train_h5'])
    train_h5_files += [config['balanced_train_h5']]

    train_dataset = H5AudioSet(train_h5_files,
                               config,
                               balanced_sampling=config['balanced_sampling'],
                               augment=True,
                               training=True)

    train_loader = DataLoader(train_dataset,
                              shuffle=True,
                              batch_size=config['batch_size'],
                              drop_last=True,
                              num_workers=config['num_workers'],
                              use_buffer_reader=True,
                              use_shared_memory=True,
                              worker_init_fn=worker_init)

    return train_loader


def get_val_loader(config):

    val_dataset = H5AudioSet([config['balanced_eval_h5']],
                             config,
                             balanced_sampling=False,
                             augment=False)

    val_loader = DataLoader(val_dataset,
                            shuffle=False,
                            batch_size=config['val_batch_size'],
                            drop_last=False,
                            num_workers=config['num_workers'])

    return val_loader


if __name__ == '__main__':
    # do some testing here
    with open('./assets/config.yaml') as f:
        config = yaml.safe_load(f)
    train_h5_files = glob.glob(config['unbalanced_train_h5'])
    dataset = H5AudioSet(train_h5_files,
                         config,
                         balanced_sampling=True,
                         augment=True,
                         training=True)
    x, y = dataset[1]
    print(x.shape, y.shape)
    dataset = H5AudioSet([config['balanced_eval_h5']],
                         config,
                         balanced_sampling=False,
                         augment=False)
    x, y = dataset[0]
    print(x.shape, y.shape)
