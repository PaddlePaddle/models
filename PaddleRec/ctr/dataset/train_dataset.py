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

import argparse
import logging
import time
import math

import paddle.fluid as fluid
import paddle.fluid.incubate.fleet.base.role_maker as role_maker
from paddle.fluid.incubate.fleet.parameter_server.distributed_transpiler import fleet
from paddle.fluid.transpiler.distribute_transpiler import DistributeTranspilerConfig

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("fluid")
logger.setLevel(logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description="PaddlePaddle CTR example")
    parser.add_argument(
        '--train_data_path',
        type=str,
        default='./data/raw/train.txt',
        help="The path of training dataset")
    parser.add_argument(
        '--test_data_path',
        type=str,
        default='./data/raw/valid.txt',
        help="The path of testing dataset")
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1000,
        help="The size of mini-batch (default:1000)")
    parser.add_argument(
        '--embedding_size',
        type=int,
        default=10,
        help="The size for embedding layer (default:10)")
    parser.add_argument(
        '--num_passes',
        type=int,
        default=10,
        help="The number of passes to train (default: 10)")
    parser.add_argument(
        '--model_output_dir',
        type=str,
        default='models',
        help='The path for model to store (default: models)')
    parser.add_argument(
        '--sparse_feature_dim',
        type=int,
        default=1000001,
        help='sparse feature hashing space for index processing')
    parser.add_argument(
        '--is_local',
        type=int,
        default=1,
        help='Local train or distributed train (default: 1)')
    parser.add_argument(
        '--cloud_train',
        type=int,
        default=0,
        help='Local train or distributed train on paddlecloud (default: 0)')
    parser.add_argument(
        '--async_mode',
        action='store_true',
        default=False,
        help='Whether start pserver in async mode to support ASGD')
    parser.add_argument(
        '--no_split_var',
        action='store_true',
        default=False,
        help='Whether split variables into blocks when update_method is pserver')
    # the following arguments is used for distributed train, if is_local == false, then you should set them
    parser.add_argument(
        '--role',
        type=str,
        default='pserver', # trainer or pserver
        help='The path for model to store (default: models)')
    parser.add_argument(
        '--endpoints',
        type=str,
        default='127.0.0.1:6000',
        help='The pserver endpoints, like: 127.0.0.1:6000,127.0.0.1:6001')
    parser.add_argument(
        '--current_endpoint',
        type=str,
        default='127.0.0.1:6000',
        help='The path for model to store (default: 127.0.0.1:6000)')
    parser.add_argument(
        '--trainer_id',
        type=int,
        default=0,
        help='The path for model to store (default: models)')
    parser.add_argument(
        '--trainers',
        type=int,
        default=1,
        help='The num of trianers, (default: 1)')
    parser.add_argument(
        '--enable_ce',
        action='store_true',
        help='If set, run the task with continuous evaluation logs.')

    return parser.parse_args()


def ctr_dnn_model(embedding_size, sparse_feature_dim):
    dense_feature_dim = 13

    def embedding_layer(input):
        return fluid.layers.embedding(
            input=input,
            is_sparse=True,
            is_distributed=False,
            size=[sparse_feature_dim, embedding_size],
            param_attr=fluid.ParamAttr(name="SparseFeatFactors",
                                       initializer=fluid.initializer.Uniform()))

    dense_input = fluid.layers.data(
        name="dense_input", shape=[dense_feature_dim], dtype='float32')

    sparse_input_ids = [
        fluid.layers.data(name="C" + str(i), shape=[1], lod_level=1, dtype='int64')
        for i in range(1, 27)]

    label = fluid.layers.data(name='label', shape=[1], dtype='int64')

    words = [dense_input] + sparse_input_ids + [label]

    sparse_embed_seq = list(map(embedding_layer, words[1:-1]))
    concated = fluid.layers.concat(sparse_embed_seq + words[0:1], axis=1)

    fc1 = fluid.layers.fc(input=concated, size=400, act='relu',
                          param_attr=fluid.ParamAttr(initializer=fluid.initializer.Normal(
                              scale=1 / math.sqrt(concated.shape[1]))))
    fc2 = fluid.layers.fc(input=fc1, size=400, act='relu',
                          param_attr=fluid.ParamAttr(initializer=fluid.initializer.Normal(
                              scale=1 / math.sqrt(fc1.shape[1]))))
    fc3 = fluid.layers.fc(input=fc2, size=400, act='relu',
                          param_attr=fluid.ParamAttr(initializer=fluid.initializer.Normal(
                              scale=1 / math.sqrt(fc2.shape[1]))))
    predict = fluid.layers.fc(input=fc3, size=2, act='softmax',
                              param_attr=fluid.ParamAttr(initializer=fluid.initializer.Normal(
                                  scale=1 / math.sqrt(fc3.shape[1]))))

    cost = fluid.layers.cross_entropy(input=predict, label=words[-1])
    avg_cost = fluid.layers.reduce_sum(cost)
    accuracy = fluid.layers.accuracy(input=predict, label=words[-1])
    auc_var, batch_auc_var, auc_states = \
        fluid.layers.auc(input=predict, label=words[-1], num_thresholds=2 ** 12, slide_steps=20)

    return avg_cost, auc_var, batch_auc_var, words


def train(args):

    avg_cost, auc_var, batch_auc_var, words = ctr_dnn_model(args.embedding_size, args.sparse_feature_dim)

    endpoints = args.endpoints.split(",")
    if args.role.upper() == "PSERVER":
        current_id = endpoints.index(args.current_endpoint)
    else:
        current_id = 0
    role = role_maker.UserDefinedRoleMaker(
        current_id=current_id,
        role=role_maker.Role.WORKER
        if args.role.upper() == "TRAINER" else role_maker.Role.SERVER,
        worker_num=args.trainers,
        server_endpoints=endpoints)

    exe = fluid.Executor(fluid.CPUPlace())
    fleet.init(role)

    strategy = DistributeTranspilerConfig()
    strategy.sync_mode = False

    optimizer = fluid.optimizer.SGD(learning_rate=0.0001)
    optimizer = fleet.distributed_optimizer(optimizer, strategy)
    optimizer.minimize(avg_cost)

    if fleet.is_server():
        logger.info("run pserver")

        fleet.init_server()
        fleet.run_server()
    elif fleet.is_worker():
        logger.info("run trainer")

        fleet.init_worker()
        exe.run(fleet.startup_program)

        thread_num = 2
        filelist = []
        for _ in range(thread_num):
            filelist.append(args.train_data_path)

        # config dataset
        dataset = fluid.DatasetFactory().create_dataset()
        dataset.set_batch_size(128)
        dataset.set_use_var(words)
        pipe_command = 'python ctr_dataset_reader.py %d %d %d %d' \
                       % args.sparse_feature_dim, args.trainer_id, 1, args.trainer_num
        dataset.set_pipe_command(pipe_command)

        dataset.set_filelist(filelist)
        dataset.set_thread(thread_num)

        for epoch_id in range(10):
            logger.info("epoch {} start".format(epoch_id))
            pass_start = time.time()
            dataset.set_filelist(filelist)
            exe.train_from_dataset(
                program=fleet.main_program,
                dataset=dataset,
                fetch_list=[avg_cost],
                fetch_info=["cost"],
                print_period=100,
                debug=False)
            pass_time = time.time() - pass_start
            logger.info("epoch {} finished, pass_time {}".format(epoch_id,
                                                                 pass_time))
        fleet.stop_worker()


if __name__ == "__main__":
    args = parse_args()
    train(args)
