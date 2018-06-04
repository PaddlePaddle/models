from __future__ import print_function

import sys
import os
import shutil
import numpy as np
import argparse
import cv2
import pdb
import paddle.v2 as paddle
import paddle.fluid as fluid
import paddle.fluid.core as core

import data_provider
from utils import convert_to_color_label
from utils import process_dir


def parse_args():
    parser = argparse.ArgumentParser('Inferring of FCN model.')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help='The number of images in a batch data. (default: %(default)d)')
    parser.add_argument(
        '--img_height',
        type=int,
        default=300,
        help='The height of input image. (default: %(default)d)')
    parser.add_argument(
        '--img_width',
        type=int,
        default=300,
        help='The width of input image. (default: %(default)d)')
    parser.add_argument(
        '--test_list',
        type=str,
        default='./data/voc2007_test.txt',
        help='The path of testing list file. (default: %(default)s)')
    parser.add_argument(
        '--vis_dir',
        type=str,
        default='demo',
        help='The path to save the tested result. (default: %(default)s)')
    parser.add_argument(
        '--data_dir',
        type=str,
        default='./data',
        help='The dataset directory. (default: %(default)s)')
    parser.add_argument(
        '--model_dir',
        type=str,
        default='./models/checkpoints',
        help='The path of saved model directory. (default: %(default)s)')
    parser.add_argument(
        '--fcn_arch',
        type=str,
        default='fcn-8s',
        help='The fcn architecture for testing, currently support : fcn-32s, fcn-16s and fcn-8s. (default: %(default)s)'
    )

    args = parser.parse_args()
    return args


def print_arguments(args):
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(vars(args).iteritems()):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')


def main(args):
    data_shape = [3, args.img_height, args.img_width]
    place = core.CUDAPlace(0)
    exe = fluid.Executor(place)

    model_dir = os.path.join(args.model_dir, '%s-model' % args.fcn_arch)
    assert (os.path.exists(model_dir))
    [inference_program, feed_target_names,
     fetch_targets] = fluid.io.load_inference_model(model_dir, exe)

    mean_value = [104, 117, 123]
    data_args = data_provider.Settings(
        data_dir=args.data_dir,
        resize_h=args.img_height,
        resize_w=args.img_width,
        mean_value=mean_value)

    infer_reader = paddle.batch(
        data_provider.infer(data_args, args.test_list),
        batch_size=args.batch_size)

    process_dir(args.vis_dir)
    for batch_id, data in enumerate(infer_reader()):
        img_data = np.array(map(lambda x: x[0], data)).astype('float32')
        img_path = np.array(map(lambda x: x[1], data))[0]
        h, w, c = cv2.imread(img_path).shape
        predict = exe.run(inference_program,
                          feed={feed_target_names[0]: img_data},
                          fetch_list=fetch_targets)
        res = np.argmax(np.squeeze(predict[0]), axis=0)
        res = convert_to_color_label(res)
        res = cv2.resize(res, (w, h), interpolation=cv2.INTER_NEAREST)
        out_img_path = os.path.join(args.vis_dir,
                                    '%s.png' % os.path.basename(img_path))
        cv2.imwrite(out_img_path, res)


if __name__ == '__main__':
    args = parse_args()
    print_arguments(args)
    main(args)
