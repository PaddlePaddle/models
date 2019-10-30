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

import sys
import os
from os.path import join, expanduser
from warnings import warn
from datetime import datetime

import matplotlib
# Force matplotlib not to use any Xwindows backend.
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib import cm

import audio
import numpy as np
from paddle import fluid
import paddle.fluid.dygraph as dg
import librosa.display
from tensorboardX import SummaryWriter

# import global hyper parameters
from hparams import hparams
from deepvoice3_paddle import frontend

_frontend = getattr(frontend, hparams.frontend)


def tts(model, text, p=0., speaker_id=None):
    """
    Convert text to speech waveform given a deepvoice3 model.

    Args:
        model (DeepVoiceTTS): Model used to synthesize waveform.
        text (str) : Input text to be synthesized
        p (float) : Replace word to pronounciation if p > 0. Default is 0.
    
    Returns:
        waveform (numpy.ndarray): Shape(T_wav, ), predicted wave form, where
            T_wav means the length of the synthesized wave form.
        alignment (numpy.ndarray): Shape(T_dec, T_enc), predicted alignment
            matrix, where T_dec means the time steps of decoder outputs, T_enc
            means the time steps of encoder outoputs.
        spectrogram (numpy.ndarray): Shape(T_lin, C_lin), predicted linear
            spectrogram, where T__lin means the time steps of linear
            spectrogram and C_lin mean sthe channels of linear spectrogram.
        mel (numpy.ndarray): Shape(T_mel, C_mel), predicted mel spectrogram,
            where T_mel means the time steps of mel spectrogram and C_mel means
            the channels of mel spectrogram.
    """
    model.eval()

    sequence = np.array(_frontend.text_to_sequence(text, p=p)).astype("int64")
    sequence = np.reshape(sequence, (1, -1, 1))
    text_positions = np.arange(1, sequence.shape[1] + 1, dtype="int64")
    text_positions = np.reshape(text_positions, (1, -1, 1))

    sequence = dg.to_variable(sequence)
    text_positions = dg.to_variable(text_positions)
    speaker_ids = None if speaker_id is None else fluid.layers.fill_constant(
        shape=[1, 1], value=speaker_id)

    # sequence: shape(1, input_length, 1)
    # text_positions: shape(1, input_length, 1)
    # Greedy decoding
    mel_outputs, linear_outputs, alignments, done = model.transduce(
        sequence, text_positions, speaker_ids)

    # reshape to the desired shape
    linear_output = linear_outputs.numpy().squeeze().T
    spectrogram = audio._denormalize(linear_output)
    alignment = alignments.numpy()[0]
    mel = mel_outputs.numpy().squeeze().T
    mel = audio._denormalize(mel)

    # Predicted audio signal
    waveform = audio.inv_spectrogram(linear_output.T)

    return waveform, alignment, spectrogram, mel


def prepare_spec_image(spectrogram):
    """
    Prepare an image from spectrogram to be written to tensorboardX 
    summary writer.
    
    Args:
        spectrogram (numpy.ndarray): Shape(T, C), spectrogram to be
            visualized, where T means the time steps of the spectrogram,
            and C means the channels of the spectrogram.
    Return:
        np.ndarray: Shape(C, T, 4), the generated image of the spectrogram,
            where T means the time steps of the spectrogram. It is treated 
            as the width of the image. And C means the channels of the
            spectrogram, which is treated as the height of the image. And 4
            means it is a 'ARGB' format.
    """

    # [0, 1]
    spectrogram = (spectrogram - np.min(spectrogram)) / (
        np.max(spectrogram) - np.min(spectrogram))
    spectrogram = np.flip(spectrogram, axis=1)  # flip against freq axis
    return np.uint8(cm.magma(spectrogram.T) * 255)


def plot_alignment(alignment, path, info=None):
    fig, ax = plt.subplots()
    im = ax.imshow(
        alignment, aspect="auto", origin="lower", interpolation="none")
    fig.colorbar(im, ax=ax)
    xlabel = "Decoder timestep"
    if info is not None:
        xlabel += "\n\n" + info
    plt.xlabel(xlabel)
    plt.ylabel("Encoder timestep")
    plt.tight_layout()
    plt.savefig(path, format="png")
    plt.close()


def time_string():
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def save_alignment(global_step, path, attn):
    plot_alignment(
        attn.T,
        path,
        info="{}, {}, step={}".format(hparams.builder,
                                      time_string(), global_step))


def eval_model(global_step, writer, model, checkpoint_dir, ismultispeaker):
    # hard coded text sequences
    texts = [
        "Scientists at the CERN laboratory say they have discovered a new particle.",
        "There's a way to measure the acute emotional intelligence that has never gone out of style.",
        "President Trump met with other leaders at the Group of 20 conference.",
        "Generative adversarial network or variational auto-encoder.",
        "Please call Stella.",
        "Some have accepted this as a miracle without any physical explanation.",
    ]

    eval_output_dir = join(checkpoint_dir, "eval")
    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)
    print("[eval] Evaluating the model, results are saved in {}".format(
        eval_output_dir))

    model.eval()
    # hard coded
    speaker_ids = [0, 1, 10] if ismultispeaker else [None]
    for speaker_id in speaker_ids:
        speaker_str = ("multispeaker{}".format(speaker_id)
                       if speaker_id is not None else "single")

        for idx, text in enumerate(texts):
            signal, alignment, _, mel = tts(model,
                                            text,
                                            p=0,
                                            speaker_id=speaker_id)
            signal /= np.max(np.abs(signal))

            # Alignment
            path = join(eval_output_dir,
                        "step{:09d}_text{}_{}_alignment.png".format(
                            global_step, idx, speaker_str))
            save_alignment(global_step, path, alignment)
            tag = "eval_averaged_alignment_{}_{}".format(idx, speaker_str)
            writer.add_image(
                tag,
                np.uint8(cm.viridis(np.flip(alignment, 1).T) * 255),
                global_step,
                dataformats='HWC')

            # Mel
            writer.add_image(
                "(Eval) Predicted mel spectrogram text{}_{}".format(
                    idx, speaker_str),
                prepare_spec_image(mel),
                global_step,
                dataformats='HWC')

            # Audio
            path = join(eval_output_dir,
                        "step{:09d}_text{}_{}_predicted.wav".format(
                            global_step, idx, speaker_str))
            audio.save_wav(signal, path)

            try:
                writer.add_audio(
                    "(Eval) Predicted audio signal {}_{}".format(idx,
                                                                 speaker_str),
                    signal,
                    global_step,
                    sample_rate=hparams.sample_rate)
            except Exception as e:
                warn(str(e))
                pass


def save_states(global_step,
                writer,
                mel_outputs,
                linear_outputs,
                attn,
                mel,
                y,
                input_lengths,
                checkpoint_dir=None):
    """
    Save states for the trainning process.
    """
    print("[train] Saving intermediate states at step {}".format(global_step))

    idx = min(1, len(input_lengths) - 1)
    input_length = input_lengths[idx]

    # Alignment, Multi-hop attention
    if attn is not None and len(attn.shape) == 4:
        attn = attn.numpy()
        for i in range(attn.shape[0]):
            alignment = attn[i]
            alignment = alignment[idx]
            tag = "alignment_layer{}".format(i + 1)
            writer.add_image(
                tag,
                np.uint8(cm.viridis(np.flip(alignment, 1).T) * 255),
                global_step,
                dataformats='HWC')

            alignment_dir = join(checkpoint_dir,
                                 "alignment_layer{}".format(i + 1))
            if not os.path.exists(alignment_dir):
                os.makedirs(alignment_dir)
            path = join(
                alignment_dir,
                "step{:09d}_layer_{}_alignment.png".format(global_step, i + 1))
            save_alignment(global_step, path, alignment)

        alignment_dir = join(checkpoint_dir, "alignment_ave")
        if not os.path.exists(alignment_dir):
            os.makedirs(alignment_dir)
        path = join(alignment_dir,
                    "step{:09d}_alignment.png".format(global_step))
        alignment = np.mean(attn, axis=0)[idx]
        save_alignment(global_step, path, alignment)

        tag = "averaged_alignment"
        writer.add_image(
            tag,
            np.uint8(cm.viridis(np.flip(alignment, 1).T) * 255),
            global_step,
            dataformats="HWC")

    if mel_outputs is not None:
        mel_output = mel_outputs[idx].numpy().squeeze().T
        mel_output = prepare_spec_image(audio._denormalize(mel_output))
        writer.add_image(
            "Predicted mel spectrogram",
            mel_output,
            global_step,
            dataformats="HWC")

    if linear_outputs is not None:
        linear_output = linear_outputs[idx].numpy().squeeze().T
        spectrogram = prepare_spec_image(audio._denormalize(linear_output))
        writer.add_image(
            "Predicted linear spectrogram",
            spectrogram,
            global_step,
            dataformats="HWC")

        signal = audio.inv_spectrogram(linear_output.T)
        signal /= np.max(np.abs(signal))
        path = join(checkpoint_dir,
                    "step{:09d}_predicted.wav".format(global_step))
        try:
            writer.add_audio(
                "Predicted audio signal",
                signal,
                global_step,
                sample_rate=hparams.sample_rate)
        except Exception as e:
            warn(str(e))
            pass
        audio.save_wav(signal, path)

    if mel_outputs is not None:
        mel_output = mel[idx].numpy().squeeze().T
        mel_output = prepare_spec_image(audio._denormalize(mel_output))
        writer.add_image(
            "Target mel spectrogram",
            mel_output,
            global_step,
            dataformats="HWC")

    if linear_outputs is not None:
        linear_output = y[idx].numpy().squeeze().T
        spectrogram = prepare_spec_image(audio._denormalize(linear_output))
        writer.add_image(
            "Target linear spectrogram",
            spectrogram,
            global_step,
            dataformats="HWC")
