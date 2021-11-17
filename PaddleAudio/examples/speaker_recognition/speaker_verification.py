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
from paddleaudio.transforms import LogMelSpectrogram
from paddleaudio.utils import Timer, get_logger
from tqdm import tqdm

logger = get_logger()

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
parser.add_argument('--device', choices=['cpu', 'gpu'], default="gpu", help="Select which device to train model, defaults to gpu.")
parser.add_argument("--batch_size", type=int, default=16, help="Total examples' number in batch for training.")
parser.add_argument("--num_workers", type=int, default=0, help="Number of workers in dataloader.")
parser.add_argument("--load_checkpoint", type=str, default='', help="Directory to load model checkpoint to contiune trainning.")
parser.add_argument("--global_embedding_norm", type=ast.literal_eval, default=True, help="Apply global normalization on speaker embeddings.")
parser.add_argument("--embedding_mean_norm", type=ast.literal_eval, default=True, help="Apply mean normalization on speaker embeddings.")
parser.add_argument("--embedding_std_norm", type=ast.literal_eval, default=False, help="Apply std normalization on speaker embeddings.")
parser.add_argument("--score_norm", type=ast.literal_eval, default=True, help="Apply score normalization.")
parser.add_argument("--norm_size", type=int, default=400000, help="Number of samples in train data used for score normalization.")
parser.add_argument("--norm_top_k", type=int, default=20000, help="Top k scores for score normalization.")
args = parser.parse_args()
# yapf: enable


def pad_right(x, target_length, mode='constant', **kwargs):
    x = np.asarray(x)
    w = target_length - len(x)
    assert w >= 0, f'Target length {target_length} is less than origin length {len(x)}'

    pad_width = [0, w]
    return np.pad(x, pad_width, mode=mode, **kwargs)


def waveform_collate_fn(batch):
    ids = [item['id'] for item in batch]
    lengths = np.asarray([item['feat'].shape[0] for item in batch])
    waveforms = list(
        map(lambda x: pad_right(x, lengths.max()),
            [item['feat'] for item in batch]))
    waveforms = np.stack(waveforms)

    # Converts into ratios.
    lengths = (lengths / lengths.max()).astype(np.float32)

    return {'ids': ids, 'waveforms': waveforms, 'lengths': lengths}


def feature_normalize(feats: paddle.Tensor,
                      lengths: paddle.Tensor,
                      mean_norm: bool = True,
                      std_norm: bool = True):

    # Features normalization if needed
    lengths = (lengths * feats.shape[-1]).astype('int64')
    for i in range(len(feats)):
        feat = feats[i, :, :lengths[i].item()]  # Excluding pad values.
        mean = feat.mean(axis=-1, keepdim=True) if mean_norm else 0
        std = feat.std(axis=-1, keepdim=True) if std_norm else 1
        feats[i, :, :lengths[i].item()] = (feat - mean) / std

    return feats


if __name__ == "__main__":
    paddle.set_device(args.device)

    feature_extractor = LogMelSpectrogram(
        sr=16000, n_fft=400, hop_length=160, n_mels=80, f_min=50)

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

    enrol_ds = VoxCeleb1(subset='enrol', random_chunk=False)
    enrol_sampler = paddle.io.BatchSampler(
        enrol_ds, batch_size=args.batch_size,
        shuffle=True)  # Shuffle to make embedding normalization more robust.
    enrol_loader = paddle.io.DataLoader(
        enrol_ds,
        batch_sampler=enrol_sampler,
        collate_fn=waveform_collate_fn,
        num_workers=args.num_workers,
        return_list=True,
    )

    test_ds = VoxCeleb1(subset='test', random_chunk=False)
    test_sampler = paddle.io.BatchSampler(
        test_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = paddle.io.DataLoader(
        test_ds,
        batch_sampler=test_sampler,
        collate_fn=waveform_collate_fn,
        num_workers=args.num_workers,
        return_list=True,
    )

    if args.score_norm:
        norm_ds = VoxCeleb1(subset='train', random_chunk=False)
        norm_sampler = paddle.io.BatchSampler(
            norm_ds, batch_size=args.batch_size, shuffle=True)
        norm_loader = paddle.io.DataLoader(
            norm_ds,
            batch_sampler=norm_sampler,
            collate_fn=waveform_collate_fn,
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
                    ids, waveforms, lengths = batch['ids'], batch[
                        'waveforms'], batch['lengths']
                    feats = feature_extractor(waveforms)  # Features extraction
                    feats = feature_normalize(
                        feats, lengths, mean_norm=True,
                        std_norm=False)  # Features normalization
                    embeddings = model.backbone(feats, lengths).squeeze(
                        -1)  # (N, emb_size, 1) -> (N, emb_size)

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
    enrol_embeddings, test_embeddings = map(
        lambda ids: paddle.stack([id2embedding[id] for id in ids]),
        [enrol_ids, test_ids])  # (N, emb_size)
    scores = cos_sim_func(enrol_embeddings, test_embeddings)

    if args.score_norm:
        n_step = args.norm_size // args.batch_size + 1  # Approximate size
        norm_data = iter(norm_loader)
        id2embedding_norm = {}
        logger.info(
            f'Computing {args.norm_size} train embeddings for score norm.')
        with paddle.no_grad():
            for i in tqdm(range(n_step)):
                batch = next(norm_data)
                ids, waveforms, lengths = batch['ids'], batch[
                    'waveforms'], batch['lengths']
                feats = feature_extractor(waveforms)
                feats = feature_normalize(
                    feats, lengths, mean_norm=True, std_norm=False)
                embeddings = model.backbone(feats, lengths).squeeze(-1)

                id2embedding_norm.update(dict(zip(ids, embeddings)))

        # Score normalization based on trainning samples.
        norm_embeddings = paddle.stack(list(id2embedding_norm.values()), axis=0)
        logger.info(f'Applying score norm...')
        for idx in tqdm(range(len(scores))):
            enrol_id, test_id = enrol_ids[idx], test_ids[idx]

            enrol_embedding, test_embedding = id2embedding[
                enrol_id], id2embedding[test_id]
            enrol_embeddings, test_embeddings = \
                map(lambda e: paddle.tile(e, [norm_embeddings.shape[0], 1]), [enrol_embedding, test_embedding])
            scores_e_norm = cos_sim_func(enrol_embeddings,
                                         norm_embeddings).topk(
                                             args.norm_top_k, axis=0)[0]
            scores_t_norm = cos_sim_func(test_embeddings, norm_embeddings).topk(
                args.norm_top_k, axis=0)[0]

            # Enrol norm
            score_e = (
                scores[idx] - paddle.mean(scores_e_norm, axis=0)) / paddle.std(
                    scores_e_norm, axis=0)
            # Test norm
            score_t = (
                scores[idx] - paddle.mean(scores_t_norm, axis=0)) / paddle.std(
                    scores_t_norm, axis=0)

            scores[idx] = (score_e + score_t) / 2

    EER, threshold = compute_eer(np.asarray(labels), scores.numpy())
    logger.info(
        f'EER of verification test: {EER*100:.4f}%, score threshold: {threshold:.5f}'
    )
