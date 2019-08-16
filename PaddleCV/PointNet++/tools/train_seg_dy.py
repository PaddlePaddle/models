#  Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

import os
import sys
import time
import shutil
import argparse
import ast
import logging
import numpy as np
import paddle.fluid as fluid
from paddle.fluid.dygraph.base import to_variable
import paddle.fluid.layers.learning_rate_scheduler as lr_scheduler

from models import *
from data.indoor3d_reader import Indoor3DReader
from utils import check_gpu, parse_outputs, Stat 

logging.root.handlers = []
FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser("PointNet++ semantic segmentation train script")
    parser.add_argument(
        '--model',
        type=str,
        default='MSG',
        help='SSG or MSG model to train, default MSG')
    parser.add_argument(
        '--use_data_parallel',
        type=ast.literal_eval,
        default=False,
        help='default training in single GPU.')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='training batch size, default 32')
    parser.add_argument(
        '--num_points',
        type=int,
        default=4096,
        help='number of points in a sample, default: 4096')
    parser.add_argument(
        '--num_classes',
        type=int,
        default=13,
        help='number of classes in dataset, default: 13')
    parser.add_argument(
        '--lr',
        type=float,
        default=0.01,
        help='initial learning rate, default 0.01')
    parser.add_argument(
        '--lr_decay',
        type=float,
        default=0.5,
        help='learning rate decay gamma, default 0.5')
    parser.add_argument(
        '--bn_momentum',
        type=float,
        default=0.9,
        help='initial batch norm momentum, default 0.9')
    parser.add_argument(
        '--bn_decay',
        type=float,
        default=0.5,
        help='batch norm momentum decay gamma, default 0.5')
    parser.add_argument(
        '--decay_steps',
        type=int,
        default=625,
        help='learning rate and batch norm momentum decay steps, default 625')
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=0.,
        help='L2 regularization weight decay coeff, default 0.')
    parser.add_argument(
        '--epoch',
        type=int,
        default=201,
        help='epoch number. default 201.')
    parser.add_argument(
        '--data_dir',
        type=str,
        default='dataset/Indoor3DSemSeg/indoor3d_sem_seg_hdf5_data',
        help='directory name to save train snapshoot')
    parser.add_argument(
        '--save_dir',
        type=str,
        default='checkpoints',
        help='directory name to save train snapshoot')
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='path to resume training based on previous checkpoints. '
        'None for not resuming any checkpoints.')
    parser.add_argument(
        '--log_interval',
        type=int,
        default=1,
        help='mini-batch interval to log.')
    args = parser.parse_args()
    return args


def exponential_with_clip(learning_rate, decay_steps, decay_rate,
                                  min_lr):
    global_step = lr_scheduler._decay_step_counter()

    lr = fluid.layers.create_global_var(
        shape=[1],
        value=float(lr),
        dtype='float32',
        persistable=False,
        name="learning_rate")

    decayed_lr = learning_rate * (decay_rate ** int(global_step / decay_steps))
    decayed_lr = max(decayed_lr, min_lr)
    fluid.layers.assign(decayed_lr, lr)

    return lr


def test(model, reader, args):
    model.eval()
    
    total_loss = 0.
    total_acc1 = 0.
    total_time = 0.
    total_sample = 0
    cur_time = time.time()
    for batch_id, data in enumerate(reader()):
        xyz_data = np.array([x[0].reshape(args.num_points, 3) for x in data]).astype('float32')
        feature_data = np.array([x[1].reshape(args.num_points, 6) for x in data]).astype('float32')
        label_data = np.array([x[2].reshape(args.num_points, 1) for x in data]).astype('int64')

        xyz = to_variable(xyz_data)
        feature = to_variable(feature_data)
        label = to_variable(label_data)
        label._stop_gradient = True

        loss, acc1 = model(xyz, feature, label)
        period = time.time() - cur_time
        cur_time = time.time()

        total_loss += loss.numpy()[0]
        total_acc1 += acc1.numpy()[0]
        total_time += period
        total_sample += 1

        if batch_id % args.log_interval == 0:
            logger.info("[TEST] batch {}, loss: {:.3f}, acc(top-1): {:.3f}, time: {:.2f}".format(batch_id, total_loss / total_sample, total_acc1 / total_sample, total_time / total_sample))
    logger.info("[TEST] finish")


def train():
    args = parse_args()
    # check whether the installed paddle is compiled with GPU
    check_gpu(True)

    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    place = fluid.CUDAPlace(fluid.dygraph.parallel.Env().dev_id) \
            if args.use_data_parallel else fluid.CUDAPlace(0)
    with fluid.dygraph.guard(place):
        model = PointNet2SemSegMSG("pointnet2_semseg_msg", num_classes=args.num_classes)
        # lr = exponential_with_clip(args.lr, args.decay_steps, args.lr_decay, 1e-5)
        lr = fluid.layers.exponential_decay(
                learning_rate=args.lr,
                decay_steps=args.decay_steps,
                decay_rate=args.lr_decay,
                staircase=True)
        optimizer = fluid.optimizer.AdamOptimizer(learning_rate=lr,
                regularization=fluid.regularizer.L2Decay(args.weight_decay))

        # def _train_reader():
        #     def reader():
        #         np.random.seed(2333)
        #         xyz = np.random.random((4096, 3)).astype('float32')
        #         feature = np.random.random((4096, 6)).astype('float32')
        #         label = np.random.uniform(0, 13, (4096, 1)).astype('int64')
        #         for i in range(10):
        #             yield [(xyz, feature, label), (xyz, feature, label)]
        #     return reader
        # train_reader = _train_reader()
        # test_reader = _train_reader()
    
        # get reader
        indoor_reader = Indoor3DReader(args.data_dir)
        train_reader = indoor_reader.get_reader(args.batch_size, args.num_points, mode='train')
        test_reader = indoor_reader.get_reader(args.batch_size, args.num_points, mode='test')

        if args.use_data_parallel:
            strategy = fluid.dygraph.parallel.prepare_context()
            model = fluid.dygraph.parallel.DataParallel(model, strategy)
            train_reader = fluid.contrib.distributed_batch_reader(train_reader)
            test_reader = fluid.contrib.distributed_batch_reader(test_reader)

        global_step = 0
        bn_momentum = args.bn_momentum
        for epoch_id in range(args.epoch):
            model.train()
            total_loss = 0.
            total_acc1 = 0.
            total_time = 0.
            total_sample = 0
            cur_time = time.time()
            for batch_id, data in enumerate(train_reader()):
                # perform bn decay
                if global_step % args.decay_steps == 0:
                    model.set_bn_momentum(bn_momentum)
                    logger.info("Set batch norm momentum as {}".format(bn_momentum))
                    bn_momentum *= args.bn_decay
                    bn_momentum = max(bn_momentum, 1e-2)
                global_step += 1

                xyz_data = np.array([x[0].reshape(args.num_points, 3) for x in data]).astype('float32')
                feature_data = np.array([x[1].reshape(args.num_points, 6) for x in data]).astype('float32')
                label_data = np.array([x[2].reshape(args.num_points, 1) for x in data]).astype('int64')

                xyz = to_variable(xyz_data)
                feature = to_variable(feature_data)
                label = to_variable(label_data)
                label._stop_gradient = True

                loss, acc1 = model(xyz, feature, label)

                if args.use_data_parallel:
                    loss = model.scale_loss(loss)
                    loss.backward()
                    model.apply_collective_grads()
                else:
                    loss.backward()
                
                optimizer.minimize(loss)
                model.clear_gradients()
                period = time.time() - cur_time
                cur_time = time.time()

                total_loss += loss.numpy()[0]
                total_acc1 += acc1.numpy()[0]
                total_time += period
                total_sample += 1
                if batch_id % args.log_interval == 0:
                    logger.info("[TRAIN] Epoch {}, batch {}, loss: {:.3f}, acc(top-1): {:.3f}, learning_rate: {:.5f} time: {:.2f}".format(epoch_id, batch_id, loss.numpy()[0], acc1.numpy()[0], optimizer._global_learning_rate().numpy()[0], period))
            fluid.dygraph.save_persistables(model.state_dict(), args.save_dir)
            test(model, test_reader, args)





if __name__ == "__main__":
    train()
