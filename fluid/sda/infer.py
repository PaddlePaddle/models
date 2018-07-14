import os
import sys
import shutil
import argparse
import numpy as np
import cv2
import shutil
import pdb

import paddle.v2 as paddle
import paddle.fluid as fluid
import paddle.fluid.core as core

from stacked_autoencoder import stacked_denoise_autoencoder
from utils import get_da_weight
from utils import normal_noise, masking_noise, salt_and_pepper_noise
from utils import mse_loss, cross_entropy_loss


def parse_args():
    parser = argparse.ArgumentParser('Training for SDA model.')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=100,
        help='The batch size in testing. (default: %(default)d)')
    parser.add_argument(
        '--img_height',
        type=int,
        default=28,
        help='The height of input image. (default: %(default)d)')
    parser.add_argument(
        '--img_width',
        type=int,
        default=28,
        help='The width of input image. (default: %(default)d)')
    parser.add_argument(
        '--gpu_id',
        type=int,
        default=0,
        help='The GPU id used to train. (default: %(default)d)')
    parser.add_argument(
        '--num_layers',
        type=list,
        default=[100, 200, 300],
        help='The number of hidden units of each layer in DAE. (default: %(default)d)'
    )
    parser.add_argument(
        '--class_num',
        type=int,
        default=10,
        help='The number of class in dataset. (default: %(default)d)')
    parser.add_argument(
        '--mode',
        type=str,
        default='SDAE',
        help='The mode of network, currently support: SDAE(stacked denoise autoencoder) and SAE(stacked regular autoencoder). (default: %(default)s)'
    )
    args = parser.parse_args()
    return args


def print_arguments(args):
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(vars(args).iteritems()):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')


def main(args):
    test_reader = paddle.batch(
        paddle.dataset.mnist.test(), batch_size=args.batch_size)

    images = fluid.layers.data(
        name='img', shape=[args.img_height * args.img_width], dtype='float32')
    sda = stacked_denoise_autoencoder(args)
    predict = sda.build_model(images)

    place = core.CUDAPlace(args.gpu_id)
    exe = fluid.Executor(place)

    model_dir = 'models'
    if args.mode == 'SDAE':
        model_dir = os.path.join(model_dir, 'SDAE')
    else:
        model_dir = os.path.join(model_dir, 'SAE')
    assert (os.path.exists(model_dir))
    [inference_program, feed_target_names,
     fetch_targets] = fluid.io.load_inference_model(model_dir, exe)

    place = core.CUDAPlace(args.gpu_id)
    exe = fluid.Executor(place)

    count_corr = 0
    count_sum = 0
    for batch_id, data in enumerate(test_reader()):
        img_data_raw = np.array(map(lambda x: x[0], data)).astype('float32')
        img_data_raw = (img_data_raw + 1) / 2

        input_label = np.array(map(lambda x: x[1], data)).astype('int64')
        predict = exe.run(inference_program,
                          feed={feed_target_names[0]: img_data_raw},
                          fetch_list=fetch_targets)
        res = np.argmax(np.squeeze(predict[0]), axis=1)
        count_sum += res.shape[0]
        count_corr += np.sum(res == input_label)
    print 'Top Accuracy : %0.3f' % (1.0 * count_corr / count_sum)


if __name__ == '__main__':
    args = parse_args()
    print_arguments(args)
    main(args)
