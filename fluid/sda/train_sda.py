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
        '--num_layers',
        type=list,
        default=[100, 200, 300],
        help='The number of hidden units of each layer in DAE. (default: %(default)d)'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.01,
        help='The learning rate. (default: %(default)f)')
    parser.add_argument(
        '--noise_ratio',
        type=float,
        default=0.25,
        help='The ratio of noise added to input data. (default: %(default)f)')
    parser.add_argument(
        '--num_epoch',
        type=int,
        default=30,
        help='The number of epochs. (default: %(default)d)')
    parser.add_argument(
        '--num_epoch_pretrain',
        type=int,
        default=20,
        help='The number of epochs used for pretrain. (default: %(default)d)')
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
    parser.add_argument(
        '--mode',
        type=str,
        default='da',
        help='The training mode, currently support: da, sda. (default: %(default)s)'
    )
    parser.add_argument(
        '--pretrain_num',
        type=int,
        default=-1,
        help='The layer number in pretrain. (default: %(default)d)')
    parser.add_argument(
        '--pretrain_strategy',
        type=str,
        default='SDAE',
        help='The pretraining strategy for building network, currently support: SDAE(stacked denoise autoencoder) and SAE(stacked regular autoencoder). (default: %(default)s)'
    )
    args = parser.parse_args()
    return args


def print_arguments(args):
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(vars(args).iteritems()):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')


def add_noise(img_data_raw, noise_type, noise_ratio):
    if noise_type == 'mask_noise':
        img_data_noise = masking_noise(img_data_raw, noise_ratio)
    elif nosie_type == 'normal':
        img_data_noise = normal_nosie(img_data_raw, noise_ratio)
    else:
        img_data_noise = img_data_raw
    return img_data_noise


def process_dir(_dir):
    if args.pretrain_num == 0:
        if os.path.exists(_dir):
            shutil.rmtree(_dir)
        os.makedirs(_dir)


def main(args):
    mnist_data_size = 60000
    train_reader = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.mnist.train(), buf_size=8192),
        batch_size=args.batch_size)
    test_reader = paddle.batch(
        paddle.dataset.mnist.test(), batch_size=args.batch_size)

    sda = stacked_denoise_autoencoder(args)

    tmp_dir = 'tmp'
    model_dir = 'models'
    process_dir(tmp_dir)
    process_dir(model_dir)
    tmp_data = os.path.join(tmp_dir, 'tmp_data.npy')

    if args.mode == 'da':
        n_hidden = args.num_layers[args.pretrain_num]
        if os.path.exists(tmp_data):
            input_data = np.load(tmp_data)
            n_visible = input_data.shape[1]
        else:
            n_visible = args.img_width * args.img_height
        input_data_next = np.zeros((mnist_data_size, n_hidden))

        images = fluid.layers.data(
            name='img_%d' % args.pretrain_num,
            shape=[n_visible],
            dtype='float32')
        label = fluid.layers.data(
            name='label_%d' % args.pretrain_num,
            shape=[n_visible],
            dtype='float32')

        predict, predict_hidden = sda.da_model(images, n_visible, n_hidden,
                                               args.pretrain_num)
        cost = cross_entropy_loss(predict, label)

        optimizer = fluid.optimizer.Adam(learning_rate=args.learning_rate)
        optimizer.minimize(cost)

        place = core.CUDAPlace(args.gpu_id)
        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())
        fluid.memory_optimize(fluid.default_main_program())

        for pass_id in range(args.num_epoch_pretrain):
            if args.pretrain_num == 0:
                for batch_id, data in enumerate(train_reader()):
                    img_data_raw = np.array(map(lambda x: x[0], data)).astype(
                        'float32')
                    img_data_raw = (img_data_raw + 1) / 2
                    img_data_noise = add_noise(img_data_raw, args.noise_type,
                                               args.noise_ratio)
                    input_label = img_data_raw
                    loss, out_hidden = exe.run(
                        fluid.default_main_program(),
                        feed={
                            'img_%d' % args.pretrain_num: img_data_noise,
                            'label_%d' % args.pretrain_num: input_label
                        },
                        fetch_list=[cost, predict_hidden])

                    input_data_next[batch_id * args.batch_size:(
                        batch_id * args.batch_size + out_hidden.shape[
                            0])] = out_hidden
                    if batch_id % 100 == 0:
                        print 'Pretrain Layer = %d, Pass = %d, batch_id = %d, Loss = %f' % (
                            args.pretrain_num, pass_id, batch_id, loss[0])
            else:
                batch_num = mnist_data_size / args.batch_size
                np.random.shuffle(input_data)
                for batch_id in range(batch_num):
                    img_data_raw = input_data[batch_id * args.batch_size:(
                        batch_id + 1) * args.batch_size].astype('float32')
                    img_data_noise = add_noise(img_data_raw, args.noise_type,
                                               args.noise_ratio)

                    input_label = img_data_raw
                    loss, out_hidden = exe.run(
                        fluid.default_main_program(),
                        feed={
                            'img_%d' % args.pretrain_num: img_data_noise,
                            'label_%d' % args.pretrain_num: input_label
                        },
                        fetch_list=[cost, predict_hidden])

                    input_data_next[batch_id * args.batch_size:(batch_id + 1) *
                                    args.batch_size] = out_hidden
                    if batch_id % 100 == 0:
                        print 'Pretrain Layer = %d, Pass = %d, batch_id = %d, Loss = %f' % (
                            args.pretrain_num, pass_id, batch_id, loss[0])

        w_name = 's%d_w' % args.pretrain_num
        b_name = 's%d_b' % args.pretrain_num
        np.save(os.path.join(tmp_dir, w_name), get_da_weight(w_name))
        np.save(os.path.join(tmp_dir, b_name), get_da_weight(b_name))
        np.save(tmp_data, input_data_next)
    elif args.mode == 'sda':
        images = fluid.layers.data(
            name='img',
            shape=[args.img_height * args.img_width],
            dtype='float32')
        label = fluid.layers.data(name='label', shape=[1], dtype='int64')
        predict = sda.build_model(images)

        cost = fluid.layers.cross_entropy(input=predict, label=label)
        cost = fluid.layers.mean(x=cost)
        acc_top1 = fluid.layers.accuracy(input=predict, label=label, k=1)

        optimizer = fluid.optimizer.Adam(learning_rate=args.learning_rate)
        optimizer.minimize(cost)

        place = core.CUDAPlace(args.gpu_id)
        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())
        fluid.memory_optimize(fluid.default_main_program())

        if args.pretrain_strategy == 'SDAE':
            for _i in range(len(args.num_layers)):
                w_name = 's%d_w' % _i
                b_name = 's%d_b' % _i
                _tensor_w = fluid.global_scope().find_var(w_name).get_tensor()
                _tensor_b = fluid.global_scope().find_var(b_name).get_tensor()
                _tensor_w.set(
                    np.load(os.path.join(tmp_dir, '%s.npy' % w_name)), place)
                _tensor_b.set(
                    np.load(os.path.join(tmp_dir, '%s.npy' % b_name)), place)

        for pass_id in range(args.num_epoch):
            for batch_id, data in enumerate(train_reader()):
                img_data_raw = np.array(map(lambda x: x[0], data)).astype(
                    'float32')
                img_data_raw = (img_data_raw + 1) / 2
                img_data_noise = add_noise(img_data_raw, args.noise_type, 0)

                input_label = np.array(map(lambda x: x[1], data)).astype(
                    'int64')
                input_label = input_label[:, np.newaxis]
                loss, recon, top1_accu = exe.run(
                    fluid.default_main_program(),
                    feed={'img': img_data_noise,
                          'label': input_label},
                    fetch_list=[cost, predict, acc_top1])
                if batch_id % 100 == 0:
                    print 'Pass = %d, batch_id = %d, Loss = %f, Accu_top1 = %f' % (
                        pass_id, batch_id, loss[0], top1_accu[0])

        if args.pretrain_strategy == 'SDAE':
            model_sub_dir = os.path.join(model_dir, 'SDAE')
        else:
            model_sub_dir = os.path.join(model_dir, 'SAE')
        fluid.io.save_inference_model(model_sub_dir, ['img'], [predict], exe)


if __name__ == '__main__':
    args = parse_args()
    print_arguments(args)
    main(args)
