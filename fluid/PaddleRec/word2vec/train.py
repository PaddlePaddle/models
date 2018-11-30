# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
train for word2vec
"""

from __future__ import print_function

import argparse
import logging
import os
import time

import numpy as np

# disable gpu training for this example
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import paddle
import paddle.fluid as fluid

import reader
from network_conf import skip_gram_word2vec

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("fluid")
logger.setLevel(logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(
        description="PaddlePaddle Word2vec example")
    parser.add_argument(
        '--train_data_path',
        type=str,
        default='./data/enwik8',
        help="The path of training dataset")
    parser.add_argument(
        '--dict_path',
        type=str,
        default='./data/enwik8_dict',
        help="The path of data dict")
    parser.add_argument(
        '--test_data_path',
        type=str,
        default='./data/text8',
        help="The path of testing dataset")
    parser.add_argument(
        '--batch_size',
        type=int,
        default=100,
        help="The size of mini-batch (default:1000)")
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
        '--embedding_size',
        type=int,
        default=64,
        help='sparse feature hashing space for index processing')

    parser.add_argument(
        '--with_hs',
        action='store_true',
        required=False,
        default=False,
        help='using hierarchical sigmoid, (default: False)')

    parser.add_argument(
        '--with_nce',
        action='store_true',
        required=False,
        default=False,
        help='using negtive sampling, (default: True)')

    parser.add_argument(
        '--max_code_length',
        type=int,
        default=40,
        help='max code length used by hierarchical sigmoid, (default: 40)')
    parser.add_argument(
        '--is_sparse',
        action='store_true',
        required=False,
        default=False,
        help='embedding and nce will use sparse or not, (default: False)')

    return parser.parse_args()


def train_loop(args, train_program, reader, py_reader, loss, trainer_id):

    train_reader = paddle.batch(
        paddle.reader.shuffle(
            reader.train((args.with_hs or (not args.with_nce))),
            buf_size=args.batch_size * 100),
        batch_size=args.batch_size)

    py_reader.decorate_paddle_reader(train_reader)

    place = fluid.CPUPlace()

    data_name_list = None

    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())
    start = time.clock()

    exec_strategy = fluid.ExecutionStrategy()

    if os.getenv("NUM_THREADS", ""):
        exec_strategy.num_threads = int(os.getenv("NUM_THREADS"))

    build_strategy = fluid.BuildStrategy()
    build_strategy.reduce_strategy = fluid.BuildStrategy.ReduceStrategy.Reduce

    train_exe = fluid.ParallelExecutor(
        use_cuda=False,
        loss_name=loss.name,
        main_program=train_program,
        build_strategy=build_strategy,
        exec_strategy=exec_strategy)

    profile_state = "CPU"
    profiler_step = 0
    profiler_step_start = 20
    profiler_step_end = 30

    for pass_id in range(args.num_passes):
        epoch_start = time.time()
        py_reader.start()
        batch_id = 0

        try:
            while True:
                if profiler_step == profiler_step_start:
                    fluid.profiler.start_profiler(profile_state)

                loss_val = train_exe.run(fetch_list=[loss.name])
                loss_val = np.mean(loss_val)

                if profiler_step == profiler_step_end:
                    fluid.profiler.stop_profiler('total', 'trainer_profile.log')
                    profiler_step += 1
                else:
                    profiler_step += 1

                if batch_id % 10 == 0:
                    logger.info("TRAIN --> pass: {} batch: {} loss: {}".format(
                        pass_id, batch_id, loss_val.mean() / args.batch_size))
                if batch_id % 100 == 0 and batch_id != 0:
                    elapsed = (time.clock() - start)
                    logger.info("Time used: {}".format(elapsed))

                if batch_id % 1000 == 0 and batch_id != 0:
                    model_dir = args.model_output_dir + '/batch-' + str(
                        batch_id)
                    if trainer_id == 0:
                        fluid.io.save_inference_model(model_dir, data_name_list,
                                                      [loss], exe)
                batch_id += 1

        except fluid.core.EOFException:
            py_reader.reset()
            epoch_end = time.time()
            print("Epoch: {0}, Train total expend: {1} ".format(
                pass_id, epoch_end - epoch_start))

            model_dir = args.model_output_dir + '/pass-' + str(pass_id)
            if trainer_id == 0:
                fluid.io.save_inference_model(model_dir, data_name_list,
                                              [loss], exe)


def train():
    args = parse_args()

    if not args.with_nce and not args.with_hs:
        logger.error("with_nce or with_hs must choose one")

    if not os.path.isdir(args.model_output_dir):
        os.mkdir(args.model_output_dir)

    word2vec_reader = reader.Word2VecReader(args.dict_path,
                                            args.train_data_path)

    logger.info("dict_size: {}".format(word2vec_reader.dict_size))

    loss, py_reader = skip_gram_word2vec(
        word2vec_reader.dict_size,
        word2vec_reader.word_frequencys,
        args.embedding_size,
        args.max_code_length,
        args.with_hs,
        args.with_nce,
        is_sparse=args.is_sparse)

    optimizer = fluid.optimizer.SGD(learning_rate=1e-3)
    optimizer.minimize(loss)

    if os.getenv("PADDLE_IS_LOCAL", "1") == "1":
        logger.info("run local training")
        main_program = fluid.default_main_program()
        train_loop(args, main_program, word2vec_reader, py_reader, loss, 0)
    else:
        logger.info("run dist training")

        trainer_id = int(os.environ["PADDLE_TRAINER_ID"])
        trainers = int(os.environ["PADDLE_TRAINERS"])
        training_role = os.environ["PADDLE_TRAINING_ROLE"]

        ports = os.getenv("PADDLE_PSERVER_PORTS", "6174")
        pserver_ip = os.getenv("PADDLE_IP", "")

        eplist = []
        for port in ports.split(","):
            eplist.append(':'.join([pserver_ip, port]))

        pserver_endpoints = ",".join(eplist)
        current_endpoint = pserver_ip + ":" + os.getenv("CUR_PORT", "2333")

        config = fluid.DistributeTranspilerConfig()
        config.slice_var_up = False
        t = fluid.DistributeTranspiler(config=config)
        t.transpile(
            trainer_id,
            pservers=pserver_endpoints,
            trainers=trainers,
            sync_mode=True)

        if training_role == "PSERVER":
            logger.info("run pserver")
            prog = t.get_pserver_program(current_endpoint)
            startup = t.get_startup_program(
                current_endpoint, pserver_program=prog)

            with open("pserver.main.proto.{}".format(os.getenv("CUR_PORT")),
                      "w") as f:
                f.write(str(prog))

            exe = fluid.Executor(fluid.CPUPlace())
            exe.run(startup)
            exe.run(prog)
        elif training_role == "TRAINER":
            logger.info("run trainer")
            train_prog = t.get_trainer_program()

            with open("trainer.main.proto.{}".format(trainer_id), "w") as f:
                f.write(str(train_prog))

            train_loop(args, train_prog, word2vec_reader, py_reader, loss,
                       trainer_id)


if __name__ == '__main__':
    train()
