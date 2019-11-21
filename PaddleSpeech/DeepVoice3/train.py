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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import io

from paddle import fluid
import paddle.fluid.dygraph as dg

from argparse import ArgumentParser
from hparams import hparams, hparams_debug_string

from nnmnkwii.datasets import FileSourceDataset
from deepvoice3_paddle.data import (TextDataSource, MelSpecDataSource,
                                    LinearSpecDataSource,
                                    PartialyRandomizedSimilarTimeLengthSampler,
                                    Dataset, make_loader, create_batch)
from deepvoice3_paddle import frontend
from deepvoice3_paddle.builder import deepvoice3, WindowRange
from deepvoice3_paddle.dry_run import dry_run
from train_model import train_model
from deepvoice3_paddle.loss import TTSLoss
from tensorboardX import SummaryWriter


def build_arg_parser():
    parser = ArgumentParser(description="Train deepvoice 3 model.")

    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Directory contains preprocessed features.")
    parser.add_argument(
        "--use-data-parallel",
        action="store_true",
        help="Whether to use data parallel training.")
    parser.add_argument(
        "--use-gpu", action="store_true", help="Whether to use gpu training.")
    parser.add_argument(
        "--output",
        type=str,
        default="result",
        help="Directory to save results")
    parser.add_argument(
        "--preset",
        type=str,
        required=True,
        help="Path of preset parameters in json format.")
    parser.add_argument(
        "--hparams",
        type=str,
        default="",
        help="Hyper parameters to override preset.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Restore model from checkpoint path if given.")
    parser.add_argument(
        "--reset-optimizer", action="store_true", help="Reset optimizer.")
    # mutually exclusive option
    train_opt = parser.add_mutually_exclusive_group()
    train_opt.add_argument(
        "--train-seq2seq-only",
        action="store_true",
        help="Train only seq2seq model")
    train_opt.add_argument(
        "--train-postnet-only",
        action="store_true",
        help="Train only postnet model.")
    parser.add_argument(
        "--speaker-id",
        type=int,
        help="Use specific speaker of data in case for multi-speaker datasets.",
    )
    return parser


def make_deepvoice3_from_hparams(hparams):
    n_vocab = getattr(frontend, hparams.frontend).n_vocab
    model = deepvoice3(
        n_vocab, hparams.text_embed_dim, hparams.num_mels,
        hparams.fft_size // 2 + 1, hparams.outputs_per_step,
        hparams.downsample_step, hparams.n_speakers, hparams.speaker_embed_dim,
        hparams.padding_idx, hparams.dropout, hparams.kernel_size,
        hparams.encoder_channels, hparams.decoder_channels,
        hparams.converter_channels, hparams.query_position_rate,
        hparams.key_position_rate, hparams.use_memory_mask,
        hparams.trainable_positional_encodings,
        hparams.force_monotonic_attention,
        hparams.use_decoder_state_for_postnet_input, hparams.max_positions,
        hparams.embedding_weight_std, hparams.speaker_embedding_weight_std,
        hparams.freeze_embedding,
        WindowRange(-hparams.window_backward, hparams.window_ahead),
        hparams.key_projection, hparams.value_projection)
    return model


def noam_learning_rate_decay(init_lr, warmup_steps=4000):
    # Noam scheme from tensor2tensor:
    warmup_steps = float(warmup_steps)
    return dg.NoamDecay(1 / (warmup_steps * (init_lr**2)), warmup_steps)


def make_optimizer_from_hparams(hparams):
    if hparams.lr_schedule is not None:
        learning_rate = noam_learning_rate_decay(hparams.initial_learning_rate,
                                                 **hparams.lr_schedule_kwargs)
    else:
        learning_rate = hparams.initial_learning_rate

    if hparams.weight_decay > 0.0:
        regularization = fluid.regularizer.L2DecayRegularizer(
            hparams.weight_decay)
    else:
        regularization = None

    optim = fluid.optimizer.Adam(
        learning_rate=learning_rate,
        beta1=hparams.adam_beta1,
        beta2=hparams.adam_beta2,
        regularization=regularization)

    if hparams.clip_thresh > 0.0:
        clipper = fluid.dygraph_grad_clip.GradClipByGlobalNorm(
            hparams.clip_thresh)
    else:
        clipper = None

    return optim, clipper


def make_loss_from_hparams(hparams):
    criterion = TTSLoss(
        hparams.masked_loss_weight, hparams.priority_freq_weight,
        hparams.binary_divergence_weight, hparams.guided_attention_sigma)
    return criterion


class MyDataParallel(dg.parallel.DataParallel):
    """
    A data parallel proxy for model.
    """

    def __init__(self, layers, strategy):
        super(MyDataParallel, self).__init__(layers, strategy)

    def __getattr__(self, key):
        if key in self.__dict__:
            return object.__getattribute__(self, key)
        elif key is "_layers":
            return object.__getattribute__(self, "_sub_layers")["_layers"]
        else:
            return getattr(
                object.__getattribute__(self, "_sub_layers")["_layers"], key)


if __name__ == "__main__":
    parser = build_arg_parser()
    args, _ = parser.parse_known_args()

    print("Command Line Args:")
    for k, v in vars(args).items():
        print("  {}: {}".format(k, v))

    # Load preset if specified
    if args.preset is not None:
        with io.open(args.preset) as f:
            hparams.parse_json(f.read())
    # Override hyper parameters
    hparams.parse(args.hparams)
    print(hparams_debug_string())

    checkpoint_dir = os.path.join(args.output, "checkpoints")
    tensorboard_dir = os.path.join(args.output, "log")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    data_root = args.data_root
    speaker_id = args.speaker_id
    X = FileSourceDataset(TextDataSource(data_root, speaker_id))
    Mel = FileSourceDataset(MelSpecDataSource(data_root, speaker_id))
    Y = FileSourceDataset(LinearSpecDataSource(data_root, speaker_id))

    frame_lengths = Mel.file_data_source.frame_lengths
    sampler = PartialyRandomizedSimilarTimeLengthSampler(
        frame_lengths, batch_size=hparams.batch_size)

    dataset = Dataset(X, Mel, Y)
    n_trainers = dg.parallel.Env().nranks
    local_rank = dg.parallel.Env().local_rank
    data_loader = make_loader(
        dataset,
        batch_size=hparams.batch_size,
        shuffle=False,
        sampler=sampler,
        create_batch_fn=create_batch,
        trainer_count=n_trainers,
        local_rank=local_rank)

    place = (fluid.CUDAPlace(dg.parallel.Env().dev_id)
             if args.use_data_parallel else fluid.CUDAPlace(0)
             if args.use_gpu else fluid.CPUPlace())
    with dg.guard(place) as g:
        pyreader = fluid.io.PyReader(capacity=10, return_list=True)
        pyreader.decorate_batch_generator(data_loader, place)

        model = make_deepvoice3_from_hparams(hparams)
        optimizer, clipper = make_optimizer_from_hparams(hparams)
        print("Log event path: {}".format(tensorboard_dir))
        writer = SummaryWriter(tensorboard_dir) if local_rank == 0 else None
        criterion = make_loss_from_hparams(hparams)

        # loading saved model
        if args.train_postnet_only or args.train_seq2seq_only:
            assert args.checkpoint is not None, \
                "you must train part of the model from a trained whole model"
        if args.train_postnet_only:
            assert hparams.use_decoder_state_for_postnet_input is False, \
                "when training only the postnet, there is no decoder states"

        if args.checkpoint is not None:
            model_dict, optimizer_dict = dg.load_dygraph(args.checkpoint)

        if args.use_data_parallel:
            strategy = dg.parallel.prepare_context()
            model = MyDataParallel(model, strategy)

        train_model(model, pyreader, criterion, optimizer, clipper, writer,
                    args, hparams)
    print("Done!")
