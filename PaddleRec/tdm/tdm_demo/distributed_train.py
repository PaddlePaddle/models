# -*- coding=utf-8 -*-
"""
#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
"""
from __future__ import print_function
import os
import time
import numpy as np
import logging
import random
from shutil import copyfile
import paddle.fluid as fluid
import paddle.fluid.incubate.fleet.base.role_maker as role_maker

from paddle.fluid.incubate.fleet.parameter_server.distribute_transpiler import fleet
from paddle.fluid.incubate.fleet.utils.hdfs import HDFSClient
from paddle.fluid.incubate.fleet.parameter_server.distribute_transpiler.distributed_strategy import StrategyFactory

from args import print_arguments, parse_args
from utils import tdm_sampler_prepare, tdm_child_prepare, tdm_emb_prepare

from train_network import TdmTrainNet


logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("fluid")
logger.setLevel(logging.INFO)

hadoop_home = os.getenv("HADOOP_HOME")
configs = {
    "fs.default.name": os.getenv("FS_NAME"),
    "hadoop.job.ugi": os.getenv("FS_UGI")
}
client = HDFSClient(hadoop_home, configs)


def get_dataset(inputs, args):
    """get dataset"""
    dataset = fluid.DatasetFactory().create_dataset()
    dataset.set_use_var(inputs)
    dataset.set_pipe_command("python ./dataset_generator.py")
    dataset.set_batch_size(args.batch_size)
    dataset.set_thread(int(args.cpu_num))
    file_list = [
        str(args.train_files_path) + "/%s" % x
        for x in os.listdir(args.train_files_path)
    ]

    # 请确保每一个训练节点都持有不同的训练文件
    # 当我们用本地多进程模拟分布式时，每个进程需要拿到不同的文件
    # 使用 fleet.split_files 可以便捷的以文件为单位分配训练样本
    if not int(args.is_cloud):
        file_list = fleet.split_files(file_list)
    logger.info("file list: {}".format(file_list))
    total_example_num = get_example_num(file_list)
    return dataset, file_list, total_example_num


def train(args):
    """run train"""
    # set random
    program = fluid.default_main_program()
    program.random_seed = args.random_seed

    # 根据环境变量确定当前机器/进程在分布式训练中扮演的角色
    # 然后使用 fleet api的 init()方法初始化这个节点
    role = role_maker.PaddleCloudRoleMaker()
    fleet.init(role)

    # 我们还可以进一步指定分布式的运行模式，通过 DistributeTranspilerConfig进行配置
    # 如下，我们设置分布式运行模式为异步(async)，同时将参数进行切分，以分配到不同的节点
    if args.sync_mode == "sync":
        strategy = StrategyFactory.create_sync_strategy()
    elif args.sync_mode == "half_async":
        strategy = StrategyFactory.create_half_async_strategy()
    elif args.sync_mode == "async":
        strategy = StrategyFactory.create_async_strategy()

    # set model
    logger.info("TDM Begin build network.")
    tdm_model = TdmTrainNet(args)
    inputs = tdm_model.input_data()

    logger.info("TDM Begin load tree travel & layer.")
    avg_cost, acc = tdm_model.tdm(inputs)
    logger.info("TDM End build network.")
    # 配置分布式的optimizer，传入我们指定的strategy，构建program
    optimizer = fluid.optimizer.AdamOptimizer(
        learning_rate=args.learning_rate, lazy_mode=True)

    optimizer = fleet.distributed_optimizer(optimizer, strategy)
    optimizer.minimize(avg_cost)
    logger.info("TDM End append backward.")

    # 根据节点角色，分别运行不同的逻辑
    if fleet.is_server():
        logger.info("TDM Run server ...")
        # 初始化及运行参数服务器节点
        logger.info("TDM init model path: {}".format(
            args.init_model_files_path))
        # 模型中除了tdm树结构相关的变量都应该在此处初始化
        fleet.init_server(args.init_model_files_path)
        lr = fluid.global_scope().find_var("learning_rate_0")
        if lr:
            lr.get_tensor().set(np.array(args.learning_rate).astype('float32'),
                                fluid.CPUPlace())
            logger.info("TDM Set learning rate {}".format(args.learning_rate))
        else:
            logger.info("TDM Didn't find learning_rate_0 param")
        logger.info("TDM load End")

        fleet.run_server()
        logger.info("TDM Run server success!")
    elif fleet.is_worker():
        logger.info("TDM Run worker ...")
        # 初始化工作节点
        fleet.init_worker()
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        logger.info("TDM Run Startup Begin")
        # 初始化含有分布式流程的fleet.startup_program
        exe.run(fleet.startup_program)

        # Set Learning Rate
        lr = fluid.global_scope().find_var("learning_rate_0")
        if lr:
            lr.get_tensor().set(np.array(args.learning_rate).astype('float32'),
                                place)
            logger.info("TDM Set learning rate {}".format(args.learning_rate))

        # Set TDM Variable
        logger.info("TDM Begin load parameter.")
        # Set TDM_Tree_Info
        # 树结构相关的变量不参与网络更新，不存储于参数服务器，因此需要在本地手动Set
        tdm_param_prepare_dict = tdm_sampler_prepare(args)
        tdm_param_prepare_dict['info_array'] = tdm_child_prepare(args)
        Numpy_model = {}
        Numpy_model['TDM_Tree_Travel'] = tdm_param_prepare_dict['travel_array']
        Numpy_model['TDM_Tree_Layer'] = tdm_param_prepare_dict['layer_array']
        Numpy_model['TDM_Tree_Info'] = tdm_param_prepare_dict['info_array']
        # Numpy_model['TDM_Tree_Emb'] = tdm_emb_prepare(args)
        # 分布式训练中，Emb存储与参数服务器，无需在本地set
        for param_name in Numpy_model:
            param_t = fluid.global_scope().find_var(param_name).get_tensor()
            param_t.set(Numpy_model[str(param_name)].astype('int32'), place)

        logger.info("TDM Run Startup End")

        # Train loop
        dataset, file_list, example_num = get_dataset(inputs, args)
        logger.info("TDM Distributed training begin ...")
        for epoch in range(args.epoch_num):
            # local shuffle
            random.shuffle(file_list)
            dataset.set_filelist(file_list)

            # 训练节点运行的是经过分布式裁剪的fleet.mian_program
            start_time = time.time()
            exe.train_from_dataset(program=fleet.main_program,
                                   dataset=dataset,
                                   fetch_list=[acc, avg_cost],
                                   fetch_info=["Epoch {} acc ".format(
                                       epoch), "Epoch {} loss ".format(epoch)],
                                   print_period=1,
                                   debug=False)
            end_time = time.time()
            logger.info("Epoch {} finished, use time {} second, speed {} example/s".format(
                epoch, end_time - start_time, example_num * 1.0 / (end_time - start_time)))

            # 默认使用0号节点保存模型
            if fleet.is_first_worker():
                model_path = os.path.join(
                    args.model_files_path, "epoch_" + str(epoch))
                fleet.save_persistables(executor=exe, dirname=model_path)
                logger.info("Begin upload files")
                # upload_files(model_path, warm_up=False)
                # 在分布式环境下时，支持上传模型到hdfs
        logger.info("TDM Before stop worker")
        fleet.stop_worker()
        logger.info("TDM Distributed training success!")


def upload_files(local_path, warm_up=False):
    """
    upload files to hdfs
    """
    remote = os.getenv("OUTPUT_PATH")
    job_id = os.getenv("SYS_JOB_ID")
    local = local_path.split('/')[-1]
    remote_path = "{}/{}/{}/{}".format(remote, job_id, "model", local)
    client.makedirs(remote_path)
    hadoop_path = "{}/".format(remote_path)

    def is_adam_param(name):
        adam_name = ['bias_beta', 'bias_moment',
                     'moment1_0', 'moment2_0', 'pow_acc']
        for i in adam_name:
            if i in name:
                return True
        return False

    if not warm_up:
        infer_model_path = os.path.join(os.getcwd(), 'infer_model_'+local)
        if not os.path.exists(infer_model_path):
            os.makedirs(infer_model_path)
        for root, _, files in os.walk(local_path):
            for f in files:
                if not is_adam_param(f):
                    copyfile(os.path.join(root, f),
                             os.path.join(infer_model_path, f))
        local_path = infer_model_path

    client.upload(hdfs_path=hadoop_path, local_path=local_path,
                  multi_processes=5, overwrite=False,
                  retry_times=3)


def get_example_num(file_list):
    """
    Count the number of samples in the file
    """
    count = 0
    for f in file_list:
        last_count = count
        for index, line in enumerate(open(f, 'r')):
            count += 1
        logger.info("file : %s has %s example" % (f, count - last_count))
    logger.info("Total example : %s" % count)
    return count


if __name__ == "__main__":
    import paddle
    paddle.enable_static()
    print(os.getcwd())
    args = parse_args()
    print_arguments(args)
    train(args)
