import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.static import InputSpec
import os
import sys
import numpy as np


def get_args(add_help=True):
    import argparse
    parser = argparse.ArgumentParser(
        description='PaddlePaddle Args', add_help=add_help)
    args = parser.parse_args()
    return args


def build_model(args):
    pass


def export(args):
    # build your own model
    model = build_model(args)

    # decorate model with jit.save
    model = paddle.jit.to_static(
        model,
        input_spec=[
            InputSpec(
                shape=[None, 3, args.img_size, args.img_size], dtype='float32')
        ])
    # save inference model
    paddle.jit.save(model, os.path.join(args.save_inference_dir, "inference"))


if __name__ == "__main__":
    args = get_args()
    export(args)
