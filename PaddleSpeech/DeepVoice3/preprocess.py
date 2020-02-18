# Part of code was adpated from https://github.com/r9y9/deepvoice3_pytorch/tree/master/preprocess.py
# Copyright (c) 2017: Ryuichi Yamamoto.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import io
import six
import os
from multiprocessing import cpu_count
from tqdm import tqdm
import importlib
from hparams import hparams, hparams_debug_string


def build_parser():
    parser = argparse.ArgumentParser(description="Data Preprocessing")
    parser.add_argument("--num-workers", type=int, help="Num workers.")
    parser.add_argument(
        "--hparams",
        type=str,
        default="",
        help="Hyper parameters to overwrite.")
    parser.add_argument(
        "--preset",
        type=str,
        required=True,
        help="Path of preset parameters (json)")
    parser.add_argument("name", type=str, help="Dataset name")
    parser.add_argument("in_dir", type=str, help="Dataset path.")
    parser.add_argument(
        "out_dir", type=str, help="Path of preprocessed dataset.")
    return parser


def preprocess(mod, in_dir, out_root, num_workers):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    metadata = mod.build_from_path(in_dir, out_dir, num_workers, tqdm=tqdm)
    write_metadata(metadata, out_dir)


def write_metadata(metadata, out_dir):
    if six.PY3:
        string_type = str
    elif six.PY2:
        string_type = unicode
    else:
        raise ValueError("Not running on Python2 or Python 3?")
    with io.open(
            os.path.join(out_dir, 'train.txt'), 'wt', encoding='utf-8') as f:
        for m in metadata:
            f.write(u'|'.join([string_type(x) for x in m]) + '\n')

    frames = sum([m[2] for m in metadata])
    frame_shift_ms = hparams.hop_size / hparams.sample_rate * 1000
    hours = frames * frame_shift_ms / (3600 * 1000)
    print('Wrote %d utterances, %d frames (%.2f hours)' %
          (len(metadata), frames, hours))
    print('Max input length:  %d' % max(len(m[3]) for m in metadata))
    print('Max output length: %d' % max(m[2] for m in metadata))


if __name__ == "__main__":
    parser = build_parser()
    args, _ = parser.parse_known_args()

    name = args.name
    in_dir = args.in_dir
    out_dir = args.out_dir
    num_workers = args.num_workers
    if num_workers is None:
        num_workers = cpu_count()
    preset = args.preset

    # Load preset if specified
    if preset is not None:
        with io.open(preset) as f:
            hparams.parse_json(f.read())
    # Override hyper parameters
    hparams.parse(args.hparams)
    assert hparams.name == "deepvoice3"
    print(hparams_debug_string())

    assert name in ["ljspeech"], "now we only supports ljspeech"
    mod = importlib.import_module(name)
    preprocess(mod, in_dir, out_dir, num_workers)
