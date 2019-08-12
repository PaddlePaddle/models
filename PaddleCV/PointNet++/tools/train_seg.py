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
        '--use_gpu',
        type=ast.literal_eval,
        default=True,
        help='default use gpu.')
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
        default=2e5,
        help='learning rate and batch norm momentum decay steps, default 2e5')
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


def train():
    args = parse_args()
    # check whether the installed paddle is compiled with GPU
    check_gpu(args.use_gpu)

    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    # build model
    startup = fluid.Program()
    train_prog = fluid.Program()
    with fluid.program_guard(train_prog, startup):
        with fluid.unique_name.guard():
            train_model = PointNet2SemSegMSG(args.num_classes, args.num_points)
            train_model.build_model()
            train_feeds = train_model.get_feeds()
            train_pyreader = train_model.get_pyreader()
            train_outputs = train_model.get_outputs()
            train_loss = train_outputs['loss']
            lr = fluid.layers.exponential_decay(
                    learning_rate=args.lr,
                    decay_steps=args.decay_steps,
                    decay_rate=args.lr_decay,
                    staircase=True)
            optimizer = fluid.optimizer.Adam(learning_rate=lr,
                    regularization=fluid.regularizer.L2Decay(args.weight_decay))
            optimizer.minimize(train_loss)
    train_keys, train_values = parse_outputs(train_outputs)

    test_prog = fluid.Program()
    with fluid.program_guard(test_prog, startup):
        with fluid.unique_name.guard():
            test_model = PointNet2SemSegMSG(args.num_classes, args.num_points)
            test_model.build_model()
            test_feeds = test_model.get_feeds()
            test_outputs = test_model.get_outputs()
            test_pyreader = test_model.get_pyreader()
    test_prog = test_prog.clone(True)
    test_keys, test_values = parse_outputs(test_outputs)

    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(startup)

    if args.resume:
        assert os.path.exists(args.resume), \
                "Given resume weight dir {} not exist.".format(args.resume)
        def if_exist(var):
            return os.path.exists(os.path.join(args.resume, var.name))
        fluid.io.load_vars(
            exe, args.resume, predicate=if_exist, main_program=train_prog)

    build_strategy = fluid.BuildStrategy()
    build_strategy.memory_optimize = False
    build_strategy.enable_inplace = False
    train_compile_prog = fluid.compiler.CompiledProgram(
            train_prog).with_data_parallel(loss_name=train_loss.name,
                    build_strategy=build_strategy)
    test_compile_prog = fluid.compiler.CompiledProgram(test_prog)

    def save_model(exe, prog, path):
        if os.path.isdir(path):
            shutil.rmtree(path)
        logger.info("Save model to {}".format(path))
        fluid.io.save_persistables(exe, path, prog)

    # def _train_reader():
    #     def reader():
    #         np.random.seed(2333)
    #         xyz = np.random.random((4096, 3)).astype('float32')
    #         feature = np.random.random((4096, 6)).astype('float32')
    #         label = np.random.uniform(0, 13, (4096, 1)).astype('int64')
    #         for i in range(10):
    #             yield [(xyz, feature, label)]
    #     return reader
    # train_reader = _train_reader()
    # test_reader = _train_reader()
    
    # get reader
    indoor_reader = Indoor3DReader(args.data_dir)
    train_reader = indoor_reader.get_reader(args.batch_size, args.num_points, mode='train')
    test_reader = indoor_reader.get_reader(args.batch_size, args.num_points, mode='test')
    train_pyreader.decorate_sample_list_generator(train_reader, place)
    test_pyreader.decorate_sample_list_generator(test_reader, place)

    train_stat = Stat()
    test_stat = Stat()
    for epoch_id in range(args.epoch):
        try:
            train_pyreader.start()
            train_iter = 0
            train_periods = []
            while True:
                cur_time = time.time()
                train_outs = exe.run(train_compile_prog, fetch_list=train_values + [lr.name])
                period = time.time() - cur_time
                train_periods.append(period)
                train_stat.update(train_keys, train_outs[:-1])
                if train_iter % args.log_interval == 0:
                    log_str = ""
                    for name, values in zip(train_keys + ['learning_rate'], train_outs):
                        log_str += "{}: {:.4f}, ".format(name, np.mean(values))
                    logger.info("[TRAIN] Epoch {}, batch {}: {}time: {:.2f}".format(epoch_id, train_iter, log_str, period))
                train_iter += 1
        except fluid.core.EOFException:
            logger.info("[TRAIN] Epoch {} finished, {}average time: {:.2f}".format(epoch_id, train_stat.get_mean_log(), np.mean(train_periods[2:])))
            save_model(exe, train_prog, os.path.join(args.save_dir, str(epoch_id)))
            train_periods = []
            
            # evaluation
            try:
                test_pyreader.start()
                test_iter = 0
                test_periods = []
                while True:
                    cur_time = time.time()
                    test_outs = exe.run(test_compile_prog, fetch_list=test_values)
                    period = time.time() - cur_time
                    test_periods.append(period)
                    test_stat.update(test_keys, test_outs)
                    if test_iter % args.log_interval == 0:
                        log_str = ""
                        for name, value in zip(test_keys, test_outs):
                            log_str += "{}: {:.4f}, ".format(name, np.mean(value))
                        logger.info("[TEST] Epoch {}, batch {}: {}time: {:.2f}".format(epoch_id, test_iter, log_str, period))
                    test_iter += 1
            except fluid.core.EOFException:
                logger.info("[TEST] Epoch {} finished, {}average time: {:.2f}".format(epoch_id, test_stat.get_mean_log(), np.mean(test_periods[2:])))
                test_periods = []

        finally:
            train_pyreader.reset()


if __name__ == "__main__":
    train()
