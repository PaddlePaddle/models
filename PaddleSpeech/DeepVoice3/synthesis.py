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

import argparse
import sys
import os
import io
from os.path import dirname, join, basename, splitext, exists
from tqdm import tqdm
import numpy as np
import nltk

from paddle import fluid
import paddle.fluid.dygraph as dg

import audio
from deepvoice3_paddle import frontend
from deepvoice3_paddle.dry_run import dry_run

from hparams import hparams
from train import make_deepvoice3_from_hparams
from eval_model import tts, plot_alignment


def build_parser():
    parser = argparse.ArgumentParser(
        description="Synthesis waveform from trained model.")
    parser.add_argument(
        "--hparams", type=str, default="", help="Hyper parameters.")
    parser.add_argument(
        "--preset",
        type=str,
        required=True,
        help="Path of preset parameters (json).")
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Whether to use gpu for generation.")
    parser.add_argument(
        "--file-name-suffix", type=str, default="", help="File name suffix.")
    parser.add_argument(
        "--max-decoder-steps", type=int, default=500, help="Max decoder steps.")
    parser.add_argument(
        "--replace_pronunciation_prob",
        type=float,
        default=0.,
        help="Probility to replace text with pronunciation.")
    parser.add_argument(
        "--speaker-id", type=int, help="Speaker ID (for multi-speaker model).")
    parser.add_argument(
        "--output-html", action="store_true", help="Output html for blog post.")
    parser.add_argument(
        "checkpoint", type=str, help="The checkpoint used for synthesis")
    parser.add_argument(
        "text_list_file",
        type=str,
        help="Text file to synthesis, a sentence per line.")
    parser.add_argument(
        "dst_dir", type=str, help="Directory to save synthesis results.")
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args, _ = parser.parse_known_args()

    checkpoint_path = args.checkpoint
    text_list_file_path = args.text_list_file
    dst_dir = args.dst_dir
    use_gpu = args.use_gpu

    max_decoder_steps = args.max_decoder_steps
    file_name_suffix = args.file_name_suffix
    replace_pronunciation_prob = args.replace_pronunciation_prob
    output_html = args.output_html
    speaker_id = args.speaker_id
    preset = args.preset

    print("Command Line Args:")
    for k, v in vars(args).items():
        print("  {}: {}".format(k, v))

    # Load preset if specified
    if preset is not None:
        with io.open(preset) as f:
            hparams.parse_json(f.read())
    # Override hyper parameters
    hparams.parse(args.hparams)
    assert hparams.name == "deepvoice3"

    place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
    with dg.guard(place):
        # Model
        model = make_deepvoice3_from_hparams(hparams)
        dry_run(model)
        model_dict, _ = dg.load_dygraph(args.checkpoint)
        model.set_dict(model_dict)

        checkpoint_name = splitext(basename(checkpoint_path))[0]

        model.seq2seq.decoder.max_decoder_steps = max_decoder_steps

        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        with io.open(text_list_file_path, "rt", encoding="utf-8") as f:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                text = line[:-1]
                words = nltk.word_tokenize(text)
                waveform, alignment, _, _ = tts(model,
                                                text,
                                                p=replace_pronunciation_prob,
                                                speaker_id=speaker_id)

                dst_wav_path = join(dst_dir, "{}_{}{}.wav".format(
                    idx, checkpoint_name, file_name_suffix))
                dst_alignment_path = join(
                    dst_dir, "{}_{}{}_alignment.png".format(
                        idx, checkpoint_name, file_name_suffix))
                plot_alignment(
                    alignment.T,
                    dst_alignment_path,
                    info="{}, {}".format(hparams.builder,
                                         basename(checkpoint_path)))
                audio.save_wav(waveform, dst_wav_path)
                name = splitext(basename(text_list_file_path))[0]
                if output_html:
                    print("""
                    {}
                    
                    ({} chars, {} words)
                    
                    <audio controls="controls" >
                    <source src="/audio/{}/{}/{}" autoplay/>
                    Your browser does not support the audio element.
                    </audio>
                    
                    <div align="center"><img src="/audio/{}/{}/{}" /></div>
                      """.format(text,
                                 len(text),
                                 len(words), hparams.builder, name,
                                 basename(dst_wav_path), hparams.builder, name,
                                 basename(dst_alignment_path)))
                else:
                    print(idx, ": {}\n ({} chars, {} words)".format(text,
                                                                    len(text),
                                                                    len(words)))

        print("Finished! Check out {} for generated audio samples.".format(
            dst_dir))
        sys.exit(0)
