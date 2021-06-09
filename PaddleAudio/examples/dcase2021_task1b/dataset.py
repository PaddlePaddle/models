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
import pickle
import subprocess
import time
import warnings

import h5py
import numpy as np
import paddle
import paddleaudio
import yaml
from paddle.io import DataLoader, Dataset, IterableDataset
from paddle.utils import download
from paddleaudio import augment
from paddleaudio.utils.log import logger

CLIP_FEATURE_URL = 'https://bj.bcebos.com/paddleaudio/examples/dcase21_task1b/clip_image_features_lp.pkl'


def get_clip_features():
    """Download pre-extracted clip features"""

    file_path = download.get_weights_path_from_url(CLIP_FEATURE_URL)
    with open(file_path, 'rb') as f:
        clip_feature = pickle.load(f)

    return clip_feature


def spect_permute(spect, tempo_axis, nblocks):
    """spectrogram  permutaion"""
    assert spect.ndim == 2, 'only supports 2d tensor or numpy array'
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
        self.clip_feature = get_clip_features()
        self.h5_files = h5_files
        self.config = config
        self.file_pointers = [h5py.File(f) for f in h5_files]
        self.all_keys, self.key2file = get_keys(self.file_pointers)
        self.augment = augment
        self.training = training
        self.balanced_sampling = balanced_sampling
        logger.info(
            f'{len(self.h5_files)} h5 files, totally {len(self.all_keys)} audio files listed'
        )
        self.labels = open(self.config['label']).read().split('\n')
        self.label2idx = {l: i for i, l in enumerate(self.labels)}
        self.key2clsidx = {
            k: self.label2idx[k.split('-')[0]]
            for k in self.all_keys
        }
        self.clsidx2key = {i: [] for i in range(len(self.labels))}
        for k in self.key2clsidx.keys():
            self.clsidx2key[self.key2clsidx[k]].append(k)

    def _process(self, x):
        assert x.shape[0] == self.config[
            'mel_bins'], 'the first dimension must be mel frequency'

        target_len = self.config['max_mel_len']
        if x.shape[1] <= target_len:
            pad_width = (target_len - x.shape[1]) // 2 + 1
            x = np.pad(x, ((0, 0), (pad_width, pad_width)))
        x = x[:, :target_len]

        if self.training and self.augment:
            x = augment.random_crop2d(
                x, self.config['mel_crop_len'], tempo_axis=1)
            x = spect_permute(x, tempo_axis=1, nblocks=random_choice([0, 2, 3]))
            aug_level = random_choice([0.2, 0.1, 0])
            x = augment.adaptive_spect_augment(x, tempo_axis=1, level=aug_level)
        return x.T

    def __getitem__(self, idx):

        if self.balanced_sampling:
            cls_id = int(np.random.randint(0, self.config['num_classes']))
            keys = self.clsidx2key[cls_id]
            k = random_choice(keys)
            cls_ids = self.key2clsidx[k]
        else:
            idx = idx % len(self.all_keys)
            k = self.all_keys[idx]
            cls_ids = self.key2clsidx[k]
        fp = self.key2file[k]
        x = fp[k][:, :]
        x = self._process(x)
        #prob = np.array(file2feature[k], 'float32')

        #y = np.zeros((self.config['num_classes'], ), 'float32')
        #y[cls_ids] = 1.0
        return x, cls_ids, self.clip_feature[k]

    def __len__(self):
        return len(self.all_keys)


def worker_init(worker_id):
    time.sleep(worker_id / 32)
    np.random.seed(int(time.time()) % 100 + worker_id)


def get_train_loader(config):
    train_h5_files = [config['train_h5']]
    train_dataset = H5AudioSet(
        train_h5_files,
        config,
        balanced_sampling=config['balanced_sampling'],
        augment=True,
        training=True)

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=config['batch_size'],
        drop_last=True,
        num_workers=config['num_workers'],
        use_buffer_reader=True,
        use_shared_memory=True,
        worker_init_fn=worker_init)

    return train_loader


def get_val_loader(config):
    val_dataset = H5AudioSet(
        [config['eval_h5']], config, balanced_sampling=False, augment=False)

    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=config['val_batch_size'],
        drop_last=False,
        num_workers=config['num_workers'])

    return val_loader


def get_test_loader(config):
    logger.info(config['test_h5'])
    test_dataset = H5AudioSet(
        [config['test_h5']], config, balanced_sampling=False, augment=False)

    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=config['val_batch_size'],
        drop_last=False,
        num_workers=config['num_workers'])

    return test_loader


if __name__ == '__main__':
    # do some testing here
    with open('./assets/config.yaml') as f:
        config = yaml.safe_load(f)
    train_h5_files = [config['train_h5']]
    dataset = H5AudioSet(
        train_h5_files,
        config,
        balanced_sampling=True,
        augment=True,
        training=True)
    x, y, p = dataset[1]
    logger.info(f'{x.shape}, {y, p.shape}')
    dataset = H5AudioSet(
        [config['eval_h5']], config, balanced_sampling=False, augment=False)
    x, y, p = dataset[0]
    logger.info(f'{x.shape}, {y, p.shape}')
