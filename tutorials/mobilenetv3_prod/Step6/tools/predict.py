import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.static import InputSpec
import os
import sys
import numpy as np
from PIL import Image

from reprod_log import ReprodLogger

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))

import paddlevision
from presets import ClassificationPresetEval


def get_args(add_help=True):
    import argparse
    parser = argparse.ArgumentParser(
        description='PaddlePaddle Classification Training', add_help=add_help)

    parser.add_argument('--model', default='mobilenet_v3_small', help='model')
    parser.add_argument('--device', default='gpu', help='device')
    parser.add_argument('--resize-size', default=224, help='resize_size')
    parser.add_argument('--crop-size', default=256, help='crop_szie')
    parser.add_argument('--img-path', default='.', help='path where to save')
    parser.add_argument('--pretrained', default=None, help='pretrained model')
    parser.add_argument('--num-classes', default=1000, help='num_classes')
    args = parser.parse_args()
    return args


@paddle.no_grad()
def main(args):
    # define model
    model = paddlevision.models.__dict__[args.model](
        pretrained=args.pretrained, num_classes=args.num_classes)

    model = nn.Sequential(model, nn.Softmax())
    model.eval()

    # define transforms
    eval_transforms = ClassificationPresetEval(args.resize_size,
                                               args.crop_size)

    with open(args.img_path, 'rb') as f:
        img = Image.open(f).convert('RGB')

    img = eval_transforms(img)
    img = paddle.to_tensor(img)
    img = img.expand([1] + img.shape)

    output = model(img).numpy()[0]

    class_id = output.argmax()
    prob = output[class_id]
    print(f"class_id: {class_id}, prob: {prob}")
    return output


if __name__ == "__main__":
    args = get_args()
    output = main(args)

    reprod_logger = ReprodLogger()
    reprod_logger.add("output", output)
    reprod_logger.save("output_training_engine.npy")
