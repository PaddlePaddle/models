from __future__ import print_function

import sys
import os
import time
import shutil
import numpy as np
import argparse
import functools
import pdb

import paddle.v2 as paddle
import paddle.fluid as fluid
import paddle.fluid.core as core

import data_provider
from vgg_fcn import vgg16_fcn
from utils import resolve_caffe_model
from utils import save_caffemodel_param
from utils import process_dir


def parse_args():
    parser = argparse.ArgumentParser('Training for FCN model.')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=8,
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
        '--class_num',
        type=int,
        default=21,
        help='The number of classes in label. (default: %(default)d)')
    parser.add_argument(
        '--gpu_id',
        type=int,
        default=0,
        help='The GPU id used to train. (default: %(default)d)')
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1e-4,
        help='The learning rate used to train. (default: %(default)f)')
    parser.add_argument(
        '--data_dir',
        type=str,
        default='./data',
        help='The dataset directory. (default: %(default)s)')
    parser.add_argument(
        '--train_list',
        type=str,
        default='./data/voc2012_trainval.txt',
        help='The path of training list file. (default: %(default)s)')
    parser.add_argument(
        '--train_list_nums',
        type=int,
        default=2913,
        help='The number of images in training list. (default: %(default)d)')
    parser.add_argument(
        '--epoch',
        type=int,
        default=60,
        help='The epoch to train. (default: %(default)d)')
    parser.add_argument(
        '--save_epoch',
        type=int,
        default=20,
        help='The per number of epoch to save the trained model. (default: %(default)d)'
    )
    parser.add_argument(
        '--save_dir',
        type=str,
        default='./models/checkpoints',
        help='The path to save the trained model. (default: %(default)s)')
    parser.add_argument(
        '--pretrain_model',
        type=str,
        default='./models/vgg16_weights',
        help='The pretrained model path used for finetuning. (default: %(default)s)'
    )
    parser.add_argument(
        '--fcn_arch',
        type=str,
        default='fcn-8s',
        help='The fcn architecture for training, currently support : fcn-32s, fcn-16s and fcn-8s. (default: %(default)s)'
    )
    args = parser.parse_args()
    return args


def print_arguments(args):
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(vars(args).iteritems()):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')


def softmax_with_cross_entropy(input, label, args):
    '''The loss function used for semantic segmentation.
    
    We convert the format of segmentation network output from (N,C,H,W) to (N*H*W, C), so the softmax_with_cross_entropy loss function in fluid
    architecture can be used directly.
    
    Args:
        input: The output of segmentation network.
        label: the corresponding label of input data.
        args: the configuration of network.
        
    Returns:
        loss: the calculated loss 
    '''
    input_transpose = fluid.layers.transpose(input, (0, 2, 3, 1))
    input_reshape = fluid.layers.reshape(
        input_transpose, shape=[-1, args.class_num])
    label = fluid.layers.reshape(label, shape=[-1, 1])

    loss = fluid.layers.softmax_with_cross_entropy(
        logits=input_reshape, label=label)
    return loss


def main(args):
    data_shape = [3, args.img_height, args.img_width]
    images = fluid.layers.data(name='img', shape=data_shape, dtype='float32')
    label = fluid.layers.data(
        name='label', shape=[args.img_height, args.img_width], dtype='int64')
    predict = vgg16_fcn(images, args)

    cost = softmax_with_cross_entropy(predict, label, args)
    avg_cost = fluid.layers.mean(x=cost)
    optimizer = fluid.optimizer.Adam(learning_rate=args.learning_rate)
    optimizer.minimize(avg_cost)

    place = core.CUDAPlace(args.gpu_id)
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())
    fluid.memory_optimize(fluid.default_main_program())

    weights_dict = resolve_caffe_model(args.pretrain_model)
    for k, v in weights_dict.items():
        _tensor = fluid.global_scope().find_var(k).get_tensor()
        _shape = np.array(_tensor).shape
        _tensor.set(v, place)

    mean_value = [104, 117, 123]
    data_args = data_provider.Settings(
        data_dir=args.data_dir,
        resize_h=args.img_height,
        resize_w=args.img_width,
        mean_value=mean_value)

    train_reader = paddle.batch(
        paddle.reader.shuffle(
            data_provider.train(data_args, args.train_list),
            buf_size=args.train_list_nums),
        batch_size=args.batch_size)

    save_dir = os.path.join(args.save_dir, args.fcn_arch)
    process_dir(save_dir)

    iters, num_samples, start_time = 0, 0, time.time()
    iters_per_epoch = args.train_list_nums / args.batch_size
    for pass_id in range(args.epoch):
        train_losses = []
        for batch_id, data in enumerate(train_reader()):
            if iters == iters_per_epoch:
                iters = 0
                break

            img_data = np.array(map(lambda x: x[0], data)).astype('float32')
            y_data = np.array(map(lambda x: x[1], data)).astype('int64')

            loss = exe.run(fluid.default_main_program(),
                           feed={'img': img_data,
                                 'label': y_data},
                           fetch_list=[avg_cost])

            iters += 1
            num_samples += len(y_data)
            print('Pass = %d/%d, Iter = %d/%d, Loss = %f' %
                  (pass_id, args.epoch, iters, iters_per_epoch, loss[0]))

        if pass_id % args.save_epoch == 0 and pass_id != 0:
            save_sub_dir = os.path.join(save_dir, str(pass_id))
            os.makedirs(save_sub_dir)
            fluid.io.save_inference_model(save_sub_dir, ['img'], [predict], exe)

        train_losses.append(loss)
        print('Pass: %d, Loss: %f' % (pass_id, np.mean(train_losses)))
        train_elapsed = time.time() - start_time
        examples_per_sec = num_samples / train_elapsed
        print('Total examples: %d, total time: %.5f, %.5f examples/sed' %
              (num_samples, train_elapsed, examples_per_sec))

    save_sub_dir = os.path.join(save_dir, 'final_model')
    fluid.io.save_inference_model(save_sub_dir, ['img'], [predict], exe)


if __name__ == '__main__':
    args = parse_args()
    print_arguments(args)
    main(args)
