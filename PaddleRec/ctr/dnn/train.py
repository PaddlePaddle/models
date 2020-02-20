#!/usr/bin/python
# -*- coding=utf-8 -*-
# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import print_function

import argparse
import logging
import os
import six
import time
import random
import numpy as np

import paddle
import paddle.fluid as fluid

from network_conf import CTR
import utils

import paddle.fluid.incubate.fleet.base.role_maker as role_maker
from paddle.fluid.incubate.fleet.parameter_server.distribute_transpiler import fleet
from paddle.fluid.transpiler.distribute_transpiler import DistributeTranspilerConfig

# disable gpu training for this example
os.environ["CUDA_VISIBLE_DEVICES"] = ""

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("fluid")
logger.setLevel(logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(
        description="PaddlePaddle CTR-DNN example")
    # -------------Data & Model Path-------------
    parser.add_argument(
        '--train_files_path',
        type=str,
        default='./train_data',
        help="The path of training dataset")
    parser.add_argument(
        '--test_files_path',
        type=str,
        default='./test_data',
        help="The path of testing dataset")
    parser.add_argument(
        '--model_path',
        type=str,
        default='models',
        help='The path for model to store (default: models)')

    # -------------Training parameter-------------
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1e-4,
        help="Initial learning rate for training")
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1000,
        help="The size of mini-batch (default:1000)")
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of epochs for training.")

    # -------------Network parameter-------------
    parser.add_argument(
        '--embedding_size',
        type=int,
        default=10,
        help="The size for embedding layer (default:10)")
    parser.add_argument(
        '--sparse_feature_dim',
        type=int,
        default=1000001,
        help='sparse feature hashing space for index processing')
    parser.add_argument(
        '--dense_feature_dim',
        type=int,
        default=13,
        help='dense feature shape')

    # -------------device parameter-------------
    parser.add_argument(
        '--is_local',
        type=int,
        default=0,
        help='Local train or distributed train (default: 1)')
    parser.add_argument(
        '--is_cloud',
        type=int,
        default=0,
        help='Local train or distributed train on paddlecloud (default: 0)')
    parser.add_argument(
        '--save_model',
        type=int,
        default=0,
        help='Save training model or not')
    parser.add_argument(
        '--cpu_num',
        type=int,
        default=2,
        help='threads for ctr training')

    return parser.parse_args()


def print_arguments(args):
    """
    print arguments
    """
    logger.info('-----------  Configuration Arguments -----------')
    for arg, value in sorted(six.iteritems(vars(args))):
        logger.info('%s: %s' % (arg, value))
    logger.info('------------------------------------------------')


def get_dataset(inputs, args):
    dataset = fluid.DatasetFactory().create_dataset()
    dataset.set_use_var(inputs)
    dataset.set_pipe_command("python dataset_generator.py")
    dataset.set_batch_size(args.batch_size)
    thread_num = int(args.cpu_num)
    dataset.set_thread(thread_num)
    file_list = [
        os.path.join(args.train_files_path, x) for x in os.listdir(args.train_files_path)
    ]
    # 请确保每一个训练节点都持有不同的训练文件
    # 当我们用本地多进程模拟分布式时，每个进程需要拿到不同的文件
    # 使用 fleet.split_files 可以便捷的以文件为单位根据节点编号分配训练样本
    if int(args.is_cloud):
        file_list = fleet.split_files(file_list)
    logger.info("file list: {}".format(file_list))

    return dataset, file_list


def local_train(args):
    # 引入模型的组网
    ctr_model = CTR()
    inputs = ctr_model.input_data(args)
    avg_cost, auc_var = ctr_model.net(inputs, args)

    # 选择反向更新优化策略
    optimizer = fluid.optimizer.Adam(args.learning_rate)
    optimizer.minimize(avg_cost)

    # 在CPU上创建训练的执行器并做参数初始化
    exe = fluid.Executor(fluid.CPUPlace())
    exe.run(fluid.default_startup_program())

    # 引入训练数据读取器与训练数据列表
    dataset, file_list = get_dataset(inputs, args)

    logger.info("Training Begin")
    for epoch in range(args.epochs):
        # 以文件为粒度进行shuffle
        random.shuffle(file_list)
        dataset.set_filelist(file_list)

        # 使用train_from_dataset实现多线程并发训练
        start_time = time.time()
        exe.train_from_dataset(program=fluid.default_main_program(),
                               dataset=dataset,
                               fetch_list=[auc_var],
                               fetch_info=["Epoch {} auc ".format(epoch)],
                               print_period=100,
                               debug=False)
        end_time = time.time()
        logger.info("epoch %d finished, use time=%d\n" %
                    ((epoch), end_time - start_time))

        if args.save_model:
            model_path = os.path.join(
                str(args.model_path), "epoch_" + str(epoch))
            if not os.path.isdir(model_path):
                os.mkdir(model_path)
            fluid.save(fluid.default_main_program(),
                       os.path.join(model_path, "checkpoint"))

    logger.info("Train Success!")


def distribute_train(args):
    # 根据环境变量确定当前机器/进程在分布式训练中扮演的角色
    # 然后使用 fleet api的 init()方法初始化这个节点
    role = role_maker.PaddleCloudRoleMaker()
    fleet.init(role)

    # 我们还可以进一步指定分布式的运行模式，通过 DistributeTranspilerConfig进行配置
    # 如下，我们设置分布式运行模式为异步(async)，同时将参数进行切分，以分配到不同的节点
    strategy = DistributeTranspilerConfig()
    strategy.sync_mode = False
    strategy.runtime_split_send_recv = True

    ctr_model = CTR()
    inputs = ctr_model.input_data(args)
    avg_cost, auc_var = ctr_model.net(inputs, args)

    # 配置分布式的optimizer，传入我们指定的strategy，构建program
    optimizer = fluid.optimizer.Adam(args.learning_rate)
    optimizer = fleet.distributed_optimizer(optimizer, strategy)
    optimizer.minimize(avg_cost)

    # 根据节点角色，分别运行不同的逻辑
    if fleet.is_server():
        # 初始化及运行参数服务器节点
        fleet.init_server()
        fleet.run_server()

    elif fleet.is_worker():
        # 初始化工作节点
        fleet.init_worker()

        exe = fluid.Executor(fluid.CPUPlace())
        # 初始化含有分布式流程的fleet.startup_program
        exe.run(fleet.startup_program)
        dataset, file_list = get_dataset(inputs, args)
        for epoch in range(args.epochs):
            # 以文件为粒度进行shuffle
            random.shuffle(file_list)
            dataset.set_filelist(file_list)

            # 训练节点运行的是经过分布式裁剪的fleet.mian_program
            start_time = time.time()
            exe.train_from_dataset(program=fleet.main_program,
                                   dataset=dataset,
                                   fetch_list=[auc_var],
                                   fetch_info=["Epoch {} auc ".format(epoch)],
                                   print_period=100,
                                   debug=False)
            end_time = time.time()
            logger.info("epoch %d finished, use time=%d\n" %
                        ((epoch), end_time - start_time))

            # 默认使用0号节点保存模型
            if args.save_model and fleet.is_first_worker():
                model_path = os.path.join(str(args.model_path), "epoch_" +
                                          str(epoch))
                fleet.save_persistables(executor=exe, dirname=model_path)

        fleet.stop_worker()
        logger.info("Distribute Train Success!")


def train():
    args = parse_args()

    if not os.path.isdir(args.model_path):
        os.mkdir(args.model_path)
    print_arguments(args)
    if args.is_cloud:
        logger.info("run cloud training")
        distribute_train(args)
    elif args.is_local:
        logger.info("run local training")
        local_train(args)


if __name__ == '__main__':
    utils.check_version()
    train()
