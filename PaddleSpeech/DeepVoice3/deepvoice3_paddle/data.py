#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
import random

import io
import platform
from os.path import dirname, join

from nnmnkwii.datasets import FileSourceDataset, FileDataSource
from os.path import join, expanduser
import random

# import global hyper parameters
from hparams import hparams
from deepvoice3_paddle import frontend, builder

_frontend = getattr(frontend, hparams.frontend)


def _pad(seq, max_len, constant_values=0):
    return np.pad(seq, (0, max_len - len(seq)),
                  mode="constant",
                  constant_values=constant_values)


def _pad_2d(x, max_len, b_pad=0):
    x = np.pad(x, [(b_pad, max_len - len(x) - b_pad), (0, 0)],
               mode="constant",
               constant_values=0)
    return x


class TextDataSource(FileDataSource):
    def __init__(self, data_root, speaker_id=None):
        self.data_root = data_root
        self.speaker_ids = None
        self.multi_speaker = False
        # If not None, filter by speaker_id
        self.speaker_id = speaker_id

    def collect_files(self):
        meta = join(self.data_root, "train.txt")
        with io.open(meta, "rt", encoding="utf-8") as f:
            lines = f.readlines()
        l = lines[0].split("|")
        assert len(l) == 4 or len(l) == 5
        self.multi_speaker = len(l) == 5
        texts = list(map(lambda l: l.split("|")[3], lines))
        if self.multi_speaker:
            speaker_ids = list(map(lambda l: int(l.split("|")[-1]), lines))
            # Filter by speaker_id
            # using multi-speaker dataset as a single speaker dataset
            if self.speaker_id is not None:
                indices = np.array(speaker_ids) == self.speaker_id
                texts = list(np.array(texts)[indices])
                self.multi_speaker = False
                return texts

            return texts, speaker_ids
        else:
            return texts

    def collect_features(self, *args):
        if self.multi_speaker:
            text, speaker_id = args
        else:
            text = args[0]
        global _frontend
        if _frontend is None:
            _frontend = getattr(frontend, hparams.frontend)
        seq = _frontend.text_to_sequence(
            text, p=hparams.replace_pronunciation_prob)

        if platform.system() == "Windows":
            if hasattr(hparams, "gc_probability"):
                _frontend = None  # memory leaking prevention in Windows
                if np.random.rand() < hparams.gc_probability:
                    gc.collect()  # garbage collection enforced
                    print("GC done")

        if self.multi_speaker:
            return np.asarray(seq, dtype=np.int32), int(speaker_id)
        else:
            return np.asarray(seq, dtype=np.int32)


class _NPYDataSource(FileDataSource):
    def __init__(self, data_root, col, speaker_id=None):
        self.data_root = data_root
        self.col = col
        self.frame_lengths = []
        self.speaker_id = speaker_id

    def collect_files(self):
        meta = join(self.data_root, "train.txt")
        with io.open(meta, "rt", encoding="utf-8") as f:
            lines = f.readlines()
        l = lines[0].split("|")
        assert len(l) == 4 or len(l) == 5
        multi_speaker = len(l) == 5
        self.frame_lengths = list(map(lambda l: int(l.split("|")[2]), lines))

        paths = list(map(lambda l: l.split("|")[self.col], lines))
        paths = list(map(lambda f: join(self.data_root, f), paths))

        if multi_speaker and self.speaker_id is not None:
            speaker_ids = list(map(lambda l: int(l.split("|")[-1]), lines))
            # Filter by speaker_id
            # using multi-speaker dataset as a single speaker dataset
            indices = np.array(speaker_ids) == self.speaker_id
            paths = list(np.array(paths)[indices])
            self.frame_lengths = list(np.array(self.frame_lengths)[indices])
            # aha, need to cast numpy.int64 to int
            self.frame_lengths = list(map(int, self.frame_lengths))

        return paths

    def collect_features(self, path):
        return np.load(path)


class MelSpecDataSource(_NPYDataSource):
    def __init__(self, data_root, speaker_id=None):
        super(MelSpecDataSource, self).__init__(data_root, 1, speaker_id)


class LinearSpecDataSource(_NPYDataSource):
    def __init__(self, data_root, speaker_id=None):
        super(LinearSpecDataSource, self).__init__(data_root, 0, speaker_id)


class PartialyRandomizedSimilarTimeLengthSampler(object):
    """Partially randmoized sampler

    1. Sort by lengths
    2. Pick a small patch and randomize it
    3. Permutate mini-batchs
    """

    def __init__(self,
                 lengths,
                 batch_size=16,
                 batch_group_size=None,
                 permutate=True):
        self.sorted_indices = np.argsort(lengths)
        self.lengths = np.array(lengths)[self.sorted_indices]
        self.batch_size = batch_size
        if batch_group_size is None:
            batch_group_size = min(batch_size * 32, len(self.lengths))
            if batch_group_size % batch_size != 0:
                batch_group_size -= batch_group_size % batch_size

        self.batch_group_size = batch_group_size
        assert batch_group_size % batch_size == 0
        self.permutate = permutate

    def __iter__(self):
        indices = self.sorted_indices.copy()
        batch_group_size = self.batch_group_size
        s, e = 0, 0
        for i in range(len(indices) // batch_group_size):
            s = i * batch_group_size
            e = s + batch_group_size
            random.shuffle(indices[s:e])

        # Permutate batches
        if self.permutate:
            perm = np.arange(len(indices[:e]) // self.batch_size)
            random.shuffle(perm)
            indices[:e] = indices[:e].reshape(
                -1, self.batch_size)[perm, :].reshape(-1)

        # Handle last elements
        s += batch_group_size
        if s < len(indices):
            random.shuffle(indices[s:])

        return iter(indices)

    def __len__(self):
        return len(self.sorted_indices)


class Dataset(object):
    def __init__(self, X, Mel, Y):
        self.X = X
        self.Mel = Mel
        self.Y = Y
        # alias
        self.multi_speaker = X.file_data_source.multi_speaker

    def __getitem__(self, idx):
        if self.multi_speaker:
            text, speaker_id = self.X[idx]
            return text, self.Mel[idx], self.Y[idx], speaker_id
        else:
            return self.X[idx], self.Mel[idx], self.Y[idx]

    def __len__(self):
        return len(self.X)


def make_loader(dataset, batch_size, shuffle, sampler, create_batch_fn,
                trainer_count, local_rank):
    assert not (
        shuffle and
        sampler), "shuffle and sampler should not be valid in the same time."
    num_samples = len(dataset)

    def wrapper():
        if sampler is None:
            ids = range(num_samples)
            if shuffle:
                random.shuffle(ids)
        else:
            ids = sampler
        batch, batches = [], []
        for idx in ids:
            batch.append(dataset[idx])
            if len(batch) >= batch_size:
                batches.append(batch)
                batch = []
            if len(batches) >= trainer_count:
                yield create_batch_fn(batches[local_rank])
                batches = []

        if len(batch) > 0:
            batches.append(batch)
        if len(batches) >= trainer_count:
            yield create_batch_fn(batches[local_rank])

    return wrapper


def create_batch(batch):
    """Create batch"""
    r = hparams.outputs_per_step
    downsample_step = hparams.downsample_step
    multi_speaker = len(batch[0]) == 4

    # Lengths
    input_lengths = [len(x[0]) for x in batch]
    max_input_len = max(input_lengths)
    input_lengths = np.array(input_lengths, dtype=np.int64)

    target_lengths = [len(x[1]) for x in batch]

    max_target_len = max(target_lengths)
    target_lengths = np.array(target_lengths, dtype=np.int64)

    if max_target_len % (r * downsample_step) != 0:
        max_target_len += (r * downsample_step) - max_target_len % (
            r * downsample_step)
        assert max_target_len % (r * downsample_step) == 0

    # Set 0 for zero beginning padding
    # imitates initial decoder states
    b_pad = r
    max_target_len += b_pad * downsample_step

    x_batch = np.array(
        [_pad(x[0], max_input_len) for x in batch], dtype=np.int64)
    x_batch = np.expand_dims(x_batch, axis=-1)

    mel_batch = np.array(
        [_pad_2d(
            x[1], max_target_len, b_pad=b_pad) for x in batch],
        dtype=np.float32)

    # down sampling is done here
    if downsample_step > 1:
        mel_batch = mel_batch[:, 0::downsample_step, :]
    mel_batch = np.expand_dims(np.transpose(mel_batch, axes=[0, 2, 1]), axis=2)

    y_batch = np.array(
        [_pad_2d(
            x[2], max_target_len, b_pad=b_pad) for x in batch],
        dtype=np.float32)
    y_batch = np.expand_dims(np.transpose(y_batch, axes=[0, 2, 1]), axis=2)

    # text positions
    text_positions = np.array(
        [_pad(np.arange(1, len(x[0]) + 1), max_input_len) for x in batch],
        dtype=np.int64)
    text_positions = np.expand_dims(text_positions, axis=-1)

    max_decoder_target_len = max_target_len // r // downsample_step

    # frame positions
    s, e = 1, max_decoder_target_len + 1
    frame_positions = np.tile(
        np.expand_dims(
            np.arange(
                s, e, dtype=np.int64), axis=0), (len(batch), 1))
    frame_positions = np.expand_dims(frame_positions, axis=-1)

    # done flags
    done = np.array([
        _pad(
            np.zeros(
                len(x[1]) // r // downsample_step - 1, dtype=np.float32),
            max_decoder_target_len,
            constant_values=1) for x in batch
    ])
    done = np.expand_dims(np.expand_dims(done, axis=1), axis=1)

    if multi_speaker:
        speaker_ids = np.expand_dims(np.array([x[3] for x in batch]), axis=-1)
        return (x_batch, input_lengths, mel_batch, y_batch, text_positions,
                frame_positions, done, target_lengths, speaker_ids)
    else:
        speaker_ids = None
        return (x_batch, input_lengths, mel_batch, y_batch, text_positions,
                frame_positions, done, target_lengths)
