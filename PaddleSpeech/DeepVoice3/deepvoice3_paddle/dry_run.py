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
from paddle import fluid
import paddle.fluid.dygraph as dg

from hparams import hparams, hparams_debug_string
from deepvoice3_paddle import frontend
from deepvoice3_paddle.deepvoice3 import DeepVoiceTTS


def dry_run(model):
    """
    Run the model once, just to get it initialized.
    """
    model.train()
    _frontend = getattr(frontend, hparams.frontend)
    batch_size = 4
    enc_length = 157
    snd_sample_length = 500

    r = hparams.outputs_per_step
    downsample_step = hparams.downsample_step
    n_speakers = hparams.n_speakers

    # make sure snd_sample_length can be divided by r * downsample_step
    linear_shift = r * downsample_step
    snd_sample_length += linear_shift - snd_sample_length % linear_shift
    decoder_length = snd_sample_length // downsample_step // r
    mel_length = snd_sample_length // downsample_step

    n_vocab = _frontend.n_vocab
    max_pos = hparams.max_positions
    spker_embed = hparams.speaker_embed_dim
    linear_dim = model.linear_dim
    mel_dim = hparams.num_mels

    x = np.random.randint(
        low=0, high=n_vocab, size=(batch_size, enc_length), dtype="int64")
    input_lengths = np.arange(
        enc_length - batch_size + 1, enc_length + 1, dtype="int64")
    mel = np.random.randn(batch_size, mel_dim, 1, mel_length).astype("float32")
    y = np.random.randn(batch_size, linear_dim, 1,
                        snd_sample_length).astype("float32")

    text_positions = np.tile(
        np.arange(
            0, enc_length, dtype="int64"), (batch_size, 1))
    text_mask = text_positions > np.expand_dims(input_lengths, 1)
    text_positions[text_mask] = 0

    frame_positions = np.tile(
        np.arange(
            1, decoder_length + 1, dtype="int64"), (batch_size, 1))

    done = np.zeros(shape=(batch_size, 1, 1, decoder_length), dtype="float32")
    target_lengths = np.array([snd_sample_length] * batch_size).astype("int64")

    speaker_ids = np.random.randint(
        low=0, high=n_speakers, size=(batch_size),
        dtype="int64") if n_speakers > 1 else None

    ismultispeaker = speaker_ids is not None

    x = dg.to_variable(x)
    input_lengths = dg.to_variable(input_lengths)
    mel = dg.to_variable(mel)
    y = dg.to_variable(y)
    text_positions = dg.to_variable(text_positions)
    frame_positions = dg.to_variable(frame_positions)
    done = dg.to_variable(done)
    target_lengths = dg.to_variable(target_lengths)
    speaker_ids = dg.to_variable(
        speaker_ids) if speaker_ids is not None else None

    # these two fields are used as numpy ndarray
    text_lengths = input_lengths.numpy()
    decoder_lengths = target_lengths.numpy() // r // downsample_step

    max_seq_len = max(text_lengths.max(), decoder_lengths.max())
    if max_seq_len >= hparams.max_positions:
        raise RuntimeError(
            "max_seq_len ({}) >= max_posision ({})\n"
            "Input text or decoder targget length exceeded the maximum length.\n"
            "Please set a larger value for ``max_position`` in hyper parameters."
            .format(max_seq_len, hparams.max_positions))

    # cause paddle's embedding layer expect shape[-1] == 1

    # first dry run runs the whole model
    mel_outputs, linear_outputs, attn, done_hat = model(
        x, input_lengths, mel, speaker_ids, text_positions, frame_positions)

    num_parameters = 0
    for k, v in model.state_dict().items():
        print("{}|{}|{}".format(k, v.shape, np.prod(v.shape)))
        num_parameters += np.prod(v.shape)
    print("now model has {} parameters".format(len(model.state_dict())))
    print("now model has {} elements".format(num_parameters))
