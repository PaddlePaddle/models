# Part of code was adpated from https://github.com/r9y9/deepvoice3_pytorch/tree/master/compute_timestamp_ratio.py
# Copyright (c) 2017: Ryuichi Yamamoto.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import io
import numpy as np
from hparams import hparams, hparams_debug_string
from deepvoice3_paddle.data import TextDataSource, MelSpecDataSource
from nnmnkwii.datasets import FileSourceDataset
from tqdm import trange
from deepvoice3_paddle import frontend


def build_parser():
    parser = argparse.ArgumentParser(
        description="Compute output/input timestamp ratio.")
    parser.add_argument(
        "--hparams", type=str, default="", help="Hyper parameters.")
    parser.add_argument(
        "--preset",
        type=str,
        required=True,
        help="Path of preset parameters (json).")
    parser.add_argument("data_root", type=str, help="path of the dataset.")
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args, _ = parser.parse_known_args()

    data_root = args.data_root
    preset = args.preset

    # Load preset if specified
    if preset is not None:
        with io.open(preset) as f:
            hparams.parse_json(f.read())
    # Override hyper parameters
    hparams.parse(args.hparams)
    assert hparams.name == "deepvoice3"

    # Code below
    X = FileSourceDataset(TextDataSource(data_root))
    Mel = FileSourceDataset(MelSpecDataSource(data_root))

    in_sizes = []
    out_sizes = []
    for i in trange(len(X)):
        x, m = X[i], Mel[i]
        if X.file_data_source.multi_speaker:
            x = x[0]
        in_sizes.append(x.shape[0])
        out_sizes.append(m.shape[0])

    in_sizes = np.array(in_sizes)
    out_sizes = np.array(out_sizes)

    input_timestamps = np.sum(in_sizes)
    output_timestamps = np.sum(
        out_sizes) / hparams.outputs_per_step / hparams.downsample_step

    print(input_timestamps, output_timestamps,
          output_timestamps / input_timestamps)
    sys.exit(0)
