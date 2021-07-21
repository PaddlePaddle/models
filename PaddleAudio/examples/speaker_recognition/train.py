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
from typing import List

import numpy as np
import paddle
import paddle.nn.functional as F
from augment import EnvCorrupt, TimeDomainSpecAugment
from loss import AdditiveAngularMargin, LogSoftmaxWrapper
from model import SpeakerClassifier
from paddleaudio.datasets import OpenRIRNoise, VoxCeleb1
from paddleaudio.models.ecapa_tdnn import ECAPA_TDNN
from paddleaudio.transforms import LogMelSpectrogram
from paddleaudio.utils import Timer, get_logger

logger = get_logger()

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
parser.add_argument('--device', choices=['cpu', 'gpu'], default="cpu", help="Select which device to train model, defaults to gpu.")
parser.add_argument("--epochs", type=int, default=50, help="Number of epoches for fine-tuning.")
parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate used to train with warmup.")
parser.add_argument("--batch_size", type=int, default=64, help="Total examples' number in batch for training.")
parser.add_argument("--num_workers", type=int, default=0, help="Number of workers in dataloader.")
parser.add_argument("--augment", type=ast.literal_eval, default=True, help="Apply audio augments.")
parser.add_argument("--checkpoint_dir", type=str, default='./checkpoint', help="Directory to save model checkpoints.")
parser.add_argument("--load_checkpoint", type=str, default='', help="Directory to load model checkpoint to contiune trainning.")
parser.add_argument("--save_freq", type=int, default=10, help="Save checkpoint every n epoch.")
parser.add_argument("--log_freq", type=int, default=10, help="Log the training infomation every n steps.")
args = parser.parse_args()
# yapf: enable


def build_augment_pipeline() -> List[paddle.nn.Layer]:
    noise_dataset = OpenRIRNoise('noise')
    rir_dataset = OpenRIRNoise('rir')

    wavedrop = TimeDomainSpecAugment(
        sample_rate=16000,
        speeds=[100],
    )
    speed_perturb = TimeDomainSpecAugment(
        sample_rate=16000,
        speeds=[95, 100, 105],
    )
    add_noise = EnvCorrupt(
        noise_dataset=noise_dataset,
        reverb_prob=0.0,
        noise_prob=1.0,
        noise_snr_low=0,
        noise_snr_high=15,
        rir_scale_factor=1.0,
    )
    add_rev = EnvCorrupt(
        rir_dataset=rir_dataset,
        reverb_prob=1.0,
        noise_prob=0.0,
        rir_scale_factor=1.0,
    )
    add_rev_noise = EnvCorrupt(
        noise_dataset=noise_dataset,
        rir_dataset=rir_dataset,
        reverb_prob=1.0,
        noise_prob=1.0,
        noise_snr_low=0,
        noise_snr_high=15,
        rir_scale_factor=1.0,
    )
    return [wavedrop, speed_perturb, add_noise, add_rev, add_rev_noise]


def waveform_augment(waveforms: paddle.Tensor,
                     augment_pipeline: List[paddle.nn.Layer]) -> paddle.Tensor:
    waveforms_aug_list = [waveforms]
    for aug in augment_pipeline:
        waveforms_aug = aug(waveforms)  # (N, L)
        if waveforms_aug.shape[1] >= waveforms.shape[1]:
            # Trunc
            waveforms_aug = waveforms_aug[:, :waveforms.shape[1]]
        else:
            # Pad
            lengths_to_pad = waveforms.shape[1] - waveforms_aug.shape[1]
            waveforms_aug = F.pad(
                waveforms_aug.unsqueeze(-1), [0, lengths_to_pad],
                data_format='NLC').squeeze(-1)
        waveforms_aug_list.append(waveforms_aug)

    return paddle.concat(waveforms_aug_list, axis=0)


def waveform_collate_fn(batch):
    waveforms = np.stack([item['feat'] for item in batch])
    labels = np.stack([item['label'] for item in batch])

    return {'waveforms': waveforms, 'labels': labels}


def feature_normalize(feats: paddle.Tensor,
                      mean_norm: bool = True,
                      std_norm: bool = True):
    # Features normalization if needed
    mean = feats.mean(axis=-1, keepdim=True) if mean_norm else 0
    std = feats.std(axis=-1, keepdim=True) if std_norm else 1
    feats = (feats - mean) / std

    return feats


if __name__ == "__main__":
    paddle.set_device(args.device)

    paddle.distributed.init_parallel_env()
    nranks = paddle.distributed.get_world_size()
    local_rank = paddle.distributed.get_rank()

    if args.augment:
        augment_pipeline = build_augment_pipeline()
    else:
        augment_pipeline = []

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
    optimizer = paddle.optimizer.AdamW(
        learning_rate=args.learning_rate, parameters=model.parameters())
    criterion = LogSoftmaxWrapper(
        loss_fn=AdditiveAngularMargin(margin=0.2, scale=30))

    start_epoch = 0
    if args.load_checkpoint:
        args.load_checkpoint = os.path.abspath(
            os.path.expanduser(args.load_checkpoint))
        try:
            # load model checkpoint
            state_dict = paddle.load(
                os.path.join(args.load_checkpoint, 'model.pdparams'))
            model.set_state_dict(state_dict)

            # load optimizer checkpoint
            state_dict = paddle.load(
                os.path.join(args.load_checkpoint, 'model.pdopt'))
            optimizer.set_state_dict(state_dict)
            if local_rank == 0:
                logger.info(f'Checkpoint loaded from {args.load_checkpoint}')
        except FileExistsError:
            if local_rank == 0:
                logger.warning('Train from scratch.')

        try:
            start_epoch = int(args.load_checkpoint[-1])
            logger.info(f'Restore training from epoch {start_epoch}.')
        except ValueError:
            pass

    train_ds = VoxCeleb1('train')
    dev_ds = VoxCeleb1('dev')

    train_sampler = paddle.io.DistributedBatchSampler(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    train_loader = paddle.io.DataLoader(
        train_ds,
        batch_sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=waveform_collate_fn,
        return_list=True,
        use_buffer_reader=True,
    )

    steps_per_epoch = len(train_sampler)
    timer = Timer(steps_per_epoch * args.epochs)
    timer.start()

    for epoch in range(start_epoch + 1, args.epochs + 1):
        model.train()

        avg_loss = 0
        num_corrects = 0
        num_samples = 0
        for batch_idx, batch in enumerate(train_loader):
            waveforms, labels = batch['waveforms'], batch['labels']

            if len(augment_pipeline) > 0:  # Waveforms augment
                waveforms = waveform_augment(waveforms, augment_pipeline)
                labels = paddle.concat(
                    [labels for i in range(len(augment_pipeline) + 1)])

            feats = feature_extractor(waveforms)  # Features extraction
            feats = feature_normalize(
                feats, mean_norm=True, std_norm=False)  # Features normalization
            logits = model(feats)

            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            if isinstance(optimizer._learning_rate,
                          paddle.optimizer.lr.LRScheduler):
                optimizer._learning_rate.step()
            optimizer.clear_grad()

            # Calculate loss
            avg_loss += loss.numpy()[0]

            # Calculate metrics
            preds = paddle.argmax(logits, axis=1)
            num_corrects += (preds == labels).numpy().sum()
            num_samples += feats.shape[0]

            timer.count()

            if (batch_idx + 1) % args.log_freq == 0 and local_rank == 0:
                lr = optimizer.get_lr()
                avg_loss /= args.log_freq
                avg_acc = num_corrects / num_samples

                print_msg = 'Epoch={}/{}, Step={}/{}'.format(
                    epoch, args.epochs, batch_idx + 1, steps_per_epoch)
                print_msg += ' loss={:.4f}'.format(avg_loss)
                print_msg += ' acc={:.4f}'.format(avg_acc)
                print_msg += ' lr={:.6f} step/sec={:.2f} | ETA {}'.format(
                    lr, timer.timing, timer.eta)
                logger.info(print_msg)

                avg_loss = 0
                num_corrects = 0
                num_samples = 0

        if epoch % args.save_freq == 0 and batch_idx + 1 == steps_per_epoch:
            if local_rank != 0:
                paddle.distributed.barrier(
                )  # Wait for valid step in main process
                continue  # Resume trainning on other process

            dev_sampler = paddle.io.BatchSampler(
                dev_ds,
                batch_size=args.batch_size // 4,
                shuffle=False,
                drop_last=False)
            dev_loader = paddle.io.DataLoader(
                dev_ds,
                batch_sampler=dev_sampler,
                collate_fn=waveform_collate_fn,
                num_workers=args.num_workers,
                return_list=True,
            )

            model.eval()
            num_corrects = 0
            num_samples = 0
            logger.info('Evaluation on validation dataset')
            with paddle.no_grad():
                for batch_idx, batch in enumerate(dev_loader):
                    waveforms, labels = batch['waveforms'], batch['labels']
                    feats = feature_extractor(waveforms)
                    feats = feature_normalize(
                        feats, mean_norm=True, std_norm=False)
                    logits = model(feats)

                    preds = paddle.argmax(logits, axis=1)
                    num_corrects += (preds == labels).numpy().sum()
                    num_samples += feats.shape[0]

            print_msg = '[Evaluation result]'
            print_msg += ' dev_acc={:.4f}'.format(num_corrects / num_samples)

            logger.info(print_msg)

            # Save model
            save_dir = os.path.join(args.checkpoint_dir,
                                    'epoch_{}'.format(epoch))
            logger.info('Saving model checkpoint to {}'.format(save_dir))
            paddle.save(model.state_dict(),
                        os.path.join(save_dir, 'model.pdparams'))
            paddle.save(optimizer.state_dict(),
                        os.path.join(save_dir, 'model.pdopt'))

            if paddle.distributed.get_world_size() > 1:
                paddle.distributed.barrier()  # Main process
