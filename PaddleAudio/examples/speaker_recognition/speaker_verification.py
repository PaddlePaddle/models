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

import argparse
import ast
import os

import numpy as np
import paddle
import paddle.nn.functional as F
from metrics import compute_eer
from model import SpeakerClassifier
from paddleaudio.datasets import VoxCeleb1
from paddleaudio.models.ecapa_tdnn import ECAPA_TDNN
from paddleaudio.utils import Timer, get_logger
from tqdm import tqdm

logger = get_logger()

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
parser.add_argument('--device', choices=['cpu', 'gpu'], default="gpu", help="Select which device to train model, defaults to gpu.")
parser.add_argument("--batch_size", type=int, default=16, help="Total examples' number in batch for training.")
parser.add_argument("--num_workers", type=int, default=0, help="Number of workers in dataloader.")
parser.add_argument("--load_checkpoint", type=str, default='', help="Directory to load model checkpoint to contiune trainning.")
parser.add_argument("--global_embedding_norm", type=bool, default=True, help="Apply global normalization on speaker embeddings.")
parser.add_argument("--embedding_mean_norm", type=bool, default=True, help="Apply mean normalization on speaker embeddings.")
parser.add_argument("--embedding_std_norm", type=bool, default=False, help="Apply std normalization on speaker embeddings.")
args = parser.parse_args()
# yapf: enable


def pad_right_2d(x, target_length, axis=-1, mode='constant', **kwargs):
    x = np.asarray(x)
    assert len(
        x.shape) == 2, f'Only 2D arrays supported, but got shape: {x.shape}'

    w = target_length - x.shape[axis]
    assert w >= 0, f'Target length {target_length} is less than origin length {x.shape[axis]}'

    if axis == 0:
        pad_width = [[0, w], [0, 0]]
    else:
        pad_width = [[0, 0], [0, w]]

    return np.pad(x, pad_width, mode=mode, **kwargs)


def feature_normalize(batch, mean_norm: bool = True, std_norm: bool = True):
    ids = [item['id'] for item in batch]
    lengths = np.asarray([item['feat'].shape[1] for item in batch])
    feats = list(
        map(lambda x: pad_right_2d(x, lengths.max()),
            [item['feat'] for item in batch]))
    feats = np.stack(feats)

    # Features normalization if needed
    for i in range(len(feats)):
        feat = feats[i][:, :lengths[i]]  # Excluding pad values.
        mean = feat.mean(axis=-1, keepdims=True) if mean_norm else 0
        std = feat.std(axis=-1, keepdims=True) if std_norm else 1
        feats[i][:, :lengths[i]] = (feat - mean) / std
        assert feats[i][:, lengths[i]:].sum(
        ) == 0  # Padding valus should all be 0.

    # Converts into ratios.
    lengths = (lengths / lengths.max()).astype(np.float32)

    return {'ids': ids, 'feats': feats, 'lengths': lengths}


if __name__ == "__main__":
    paddle.set_device(args.device)

    model_conf = {
        "input_size": 80,
        "channels": [1024, 1024, 1024, 1024, 3072],
        "kernel_sizes": [5, 3, 3, 3, 1],
        "dilations": [1, 2, 3, 4, 1],
        "attention_channels": 128,
        "lin_neurons": 192,
    }
    ecapa_tdnn = ECAPA_TDNN(**model_conf)
    model = SpeakerClassifier(
        backbone=ecapa_tdnn, num_class=VoxCeleb1.num_speakers)

    args.load_checkpoint = os.path.abspath(
        os.path.expanduser(args.load_checkpoint))

    # load model checkpoint
    state_dict = paddle.load(
        os.path.join(args.load_checkpoint, 'model.pdparams'))
    model.set_state_dict(state_dict)
    logger.info(f'Checkpoint loaded from {args.load_checkpoint}')

    enrol_ds = VoxCeleb1(
        subset='enrol',
        feat_type='melspectrogram',
        random_chunk=False,
        n_mels=80,
        window_size=400,
        hop_length=160)
    enrol_sampler = paddle.io.BatchSampler(
        enrol_ds, batch_size=args.batch_size,
        shuffle=True)  # Shuffle to make embedding normalization more robust.
    enrol_loader = paddle.io.DataLoader(
        enrol_ds,
        batch_sampler=enrol_sampler,
        collate_fn=lambda x: feature_normalize(
            x, mean_norm=True, std_norm=False),
        num_workers=args.num_workers,
        return_list=True,
    )

    test_ds = VoxCeleb1(
        subset='test',
        feat_type='melspectrogram',
        random_chunk=False,
        n_mels=80,
        window_size=400,
        hop_length=160)

    test_sampler = paddle.io.BatchSampler(
        test_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = paddle.io.DataLoader(
        test_ds,
        batch_sampler=test_sampler,
        collate_fn=lambda x: feature_normalize(
            x, mean_norm=True, std_norm=False),
        num_workers=args.num_workers,
        return_list=True,
    )

    # Compute embeddings of audios in enrol and test dataset from model.
    model.eval()

    if args.global_embedding_norm:
        embedding_mean = None
        embedding_std = None
        mean_norm = args.embedding_mean_norm
        std_norm = args.embedding_std_norm
        batch_count = 0

    id2embedding = {}
    # Run multi times to make embedding normalization more stable.
    for i in range(2):
        for dl in [enrol_loader, test_loader]:
            logger.info(
                f'Loop {[i+1]}: Computing embeddings on {dl.dataset.subset} dataset'
            )
            with paddle.no_grad():
                for batch_idx, batch in enumerate(tqdm(dl)):
                    ids, feats, lengths = batch['ids'], batch['feats'], batch[
                        'lengths']
                    embeddings = model.backbone(feats, lengths).squeeze(
                        -1).numpy()  # (N, emb_size, 1) -> (N, emb_size)

                    # Global embedding normalization.
                    if args.global_embedding_norm:
                        batch_count += 1
                        mean = embeddings.mean(axis=0) if mean_norm else 0
                        std = embeddings.std(axis=0) if std_norm else 1
                        # Update global mean and std.
                        if embedding_mean is None and embedding_std is None:
                            embedding_mean, embedding_std = mean, std
                        else:
                            weight = 1 / batch_count  # Weight decay by batches.
                            embedding_mean = (
                                1 - weight) * embedding_mean + weight * mean
                            embedding_std = (
                                1 - weight) * embedding_std + weight * std
                        # Apply global embedding normalization.
                        embeddings = (
                            embeddings - embedding_mean) / embedding_std

                    # Update embedding dict.
                    id2embedding.update(dict(zip(ids, embeddings)))

    # Compute cosine scores.
    labels = []
    enrol_ids = []
    test_ids = []
    with open(VoxCeleb1.veri_test_file, 'r') as f:
        for line in f.readlines():
            label, enrol_id, test_id = line.strip().split(' ')
            labels.append(int(label))
            enrol_ids.append(enrol_id.split('.')[0].replace('/', '-'))
            test_ids.append(test_id.split('.')[0].replace('/', '-'))

    cos_sim_func = paddle.nn.CosineSimilarity(axis=1)
    enrol_embeddings, test_embeddings = map(lambda ids: paddle.to_tensor(
        np.asarray([id2embedding[id] for id in ids], dtype='float32')),
                                            [enrol_ids, test_ids
                                             ])  # (N, emb_size)
    scores = cos_sim_func(enrol_embeddings, test_embeddings)
    EER, threshold = compute_eer(np.asarray(labels), scores.numpy())
    logger.info(
        f'EER of verification test: {EER*100:.4f}%, score threshold: {threshold:.5f}'
    )
