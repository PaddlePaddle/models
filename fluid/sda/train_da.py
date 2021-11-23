import os
import sys
import shutil
import argparse
import numpy as np
import cv2
import pdb

import paddle.v2 as paddle
import paddle.fluid as fluid
import paddle.fluid.core as core

from autoencoder import denoise_autoencoder
from utils import get_da_weight
from utils import process_dir
from utils import vis_weight
from utils import normal_noise, masking_noise, salt_and_pepper_noise
from utils import mse_loss, cross_entropy_loss


def parse_args():
    parser = argparse.ArgumentParser('Training denoise autoencoder(DA) model.')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=128,
        help='The number of batch size for training. (default: %(default)d)')
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
        '--n_hidden',
        type=int,
        default=144,
        help='The number of hidden units in DA. (default: %(default)d)')
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.01,
        help='The learning rate. (default: %(default)f)')
    parser.add_argument(
        '--nosie_ratio',
        type=float,
        default=0.50,
        help='The ratio of noise added to input data. (default: %(default)f)')
    parser.add_argument(
        '--num_epoch',
        type=int,
        default=30,
        help='The number of epochs. (default: %(default)d)')
    parser.add_argument(
        '--class_num',
        type=int,
        default=10,
        help='The number of class in dataset. (default: %(default)d)')
    parser.add_argument(
        '--noise_type',
        type=str,
        default='mask_noise',
        help='The nosie type used in corruption, currently support: normal, mask_noise, salt_pepper_noise. (default: %(default)s)'
    )
    args = parser.parse_args()
    return args


def print_arguments(args):
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(vars(args).iteritems()):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')


def main(args):
    images = fluid.layers.data(
        name='img', shape=[args.img_height * args.img_width], dtype='float32')
    label = fluid.layers.data(
        name='label', shape=[args.img_height * args.img_width], dtype='float32')
    predict = denoise_autoencoder(images, args)
    cost = cross_entropy_loss(predict, label)

    optimizer = fluid.optimizer.Adam(learning_rate=args.learning_rate)
    optimizer.minimize(cost)

    place = core.CUDAPlace(args.gpu_id)
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())
    fluid.memory_optimize(fluid.default_main_program())

    train_reader = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.mnist.train(), buf_size=8192),
        batch_size=args.batch_size)
    test_reader = paddle.batch(
        paddle.dataset.mnist.test(), batch_size=args.batch_size)

    for pass_id in range(args.num_epoch):
        for batch_id, data in enumerate(train_reader()):
            img_data_raw = np.array(map(lambda x: x[0], data)).astype('float32')
            img_data_raw = (img_data_raw + 1) / 2
            if args.noise_type == 'normal':
                img_data_noise = normal_noise(img_data_raw, args.nosie_ratio)
            elif args.noise_type == 'mask_noise':
                img_data_noise = masking_noise(img_data_raw, args.nosie_ratio)
            else:
                img_data_noise = img_data_raw

            input_label = img_data_raw
            loss, recon = exe.run(
                fluid.default_main_program(),
                feed={'img': img_data_noise,
                      'label': input_label},
                fetch_list=[cost, predict])
            if batch_id % 100 == 0:
                print 'Pass = %d, batch_id = %d, Loss = %f' % (
                    pass_id, batch_id, loss[0])
    weight = get_da_weight('W')
    vis_weight(weight, args.nosie_ratio)


if __name__ == '__main__':
    args = parse_args()
    print_arguments(args)
    main(args)
