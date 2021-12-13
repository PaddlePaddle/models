import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.static import InputSpec
import os
import sys
import numpy as np

paddle.set_device("cpu")

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))

import paddlevision


def get_args(add_help=True):
    import argparse
    parser = argparse.ArgumentParser(
        description='PaddlePaddle Classification Training', add_help=add_help)

    parser.add_argument('--model', default='mobilenet_v3_small', help='model')
    parser.add_argument('--device', default='gpu', help='device')
    parser.add_argument('--img-size', default=224, help='image size to export')
    parser.add_argument(
        '--save-inference-dir', default='.', help='path where to save')
    parser.add_argument('--pretrained', default=None, help='pretrained model')
    parser.add_argument('--num-classes', default=1000, help='num_classes')

    args = parser.parse_args()
    return args


def export(args):
    model = paddlevision.models.__dict__[args.model](
        pretrained=args.pretrained, num_classes=args.num_classes)
    model = nn.Sequential(model, nn.Softmax())
    model.eval()

    model = paddle.jit.to_static(
        model,
        input_spec=[
            InputSpec(
                shape=[None, 3, args.img_size, args.img_size], dtype='float32')
        ])
    paddle.jit.save(model, os.path.join(args.save_inference_dir, "inference"))
    print(f"inference model has been saved into {args.save_inference_dir}")


if __name__ == "__main__":
    args = get_args()
    export(args)
