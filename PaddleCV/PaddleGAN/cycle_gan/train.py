#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os


def set_paddle_flags(flags):
    for key, value in flags.items():
        if os.environ.get(key, None) is None:
            os.environ[key] = str(value)


use_cudnn_deterministic = os.environ.get('FLAGS_cudnn_deterministic', None)

if use_cudnn_deterministic:
    use_cudnn_exhaustive_search = 0
else:
    use_cudnn_exhaustive_search = 1

# NOTE(paddle-dev): All of these flags should be
# set before `import paddle`. Otherwise, it would
# not take any effect. 
set_paddle_flags({
    'FLAGS_cudnn_exhaustive_search': use_cudnn_exhaustive_search,
    'FLAGS_conv_workspace_size_limit': 256,
    'FLAGS_eager_delete_tensor_gb': 0,  # enable gc 
    # You can omit the following settings, because the default
    # value of FLAGS_memory_fraction_of_eager_deletion is 1,
    # and default value of FLAGS_fast_eager_deletion_mode is 1 
    'FLAGS_memory_fraction_of_eager_deletion': 1,
    'FLAGS_fast_eager_deletion_mode': 1
})

import random
import sys
import paddle
import argparse
import functools
import time
import numpy as np
from scipy.misc import imsave
import paddle.fluid as fluid
import paddle.fluid.profiler as profiler
import data_reader
from utility import add_arguments, print_arguments, ImagePool
from trainer import GATrainer, GBTrainer, DATrainer, DBTrainer

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('batch_size',        int,   1,          "Minibatch size.")
add_arg('epoch',             int,   2,        "The number of epoched to be trained.")
add_arg('output',            str,   "./output", "The directory the model and the test result to be saved to.")
add_arg('init_model',        str,   None,       "The init model file of directory.")
add_arg('save_checkpoints',  bool,  True,       "Whether to save checkpoints.")
add_arg('run_test',          bool,  True,       "Whether to run test.")
add_arg('use_gpu',           bool,  True,       "Whether to use GPU to train.")
add_arg('profile',           bool,  False,       "Whether to profile.")
add_arg('run_ce',            bool,  False,       "Whether to run for model ce.")
# yapf: enable


def train(args):

    max_images_num = data_reader.max_images_num()
    shuffle = True
    if args.run_ce:
        np.random.seed(10)
        fluid.default_startup_program().random_seed = 90
        max_images_num = 1
        shuffle = False
    data_shape = [-1] + data_reader.image_shape()

    input_A = fluid.layers.data(
        name='input_A', shape=data_shape, dtype='float32')
    input_B = fluid.layers.data(
        name='input_B', shape=data_shape, dtype='float32')
    fake_pool_A = fluid.layers.data(
        name='fake_pool_A', shape=data_shape, dtype='float32')
    fake_pool_B = fluid.layers.data(
        name='fake_pool_B', shape=data_shape, dtype='float32')

    g_A_trainer = GATrainer(input_A, input_B)
    g_B_trainer = GBTrainer(input_A, input_B)
    d_A_trainer = DATrainer(input_A, fake_pool_A)
    d_B_trainer = DBTrainer(input_B, fake_pool_B)

    # prepare environment
    place = fluid.CPUPlace()
    if args.use_gpu:
        place = fluid.CUDAPlace(0)
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())
    A_pool = ImagePool()
    B_pool = ImagePool()

    A_reader = paddle.batch(
        data_reader.a_reader(shuffle=shuffle), args.batch_size)()
    B_reader = paddle.batch(
        data_reader.b_reader(shuffle=shuffle), args.batch_size)()
    if not args.run_ce:
        A_test_reader = data_reader.a_test_reader()
        B_test_reader = data_reader.b_test_reader()

    def test(epoch):
        out_path = args.output + "/test"
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        i = 0
        for data_A, data_B in zip(A_test_reader(), B_test_reader()):
            A_name = data_A[1]
            B_name = data_B[1]
            tensor_A = fluid.LoDTensor()
            tensor_B = fluid.LoDTensor()
            tensor_A.set(data_A[0], place)
            tensor_B.set(data_B[0], place)
            fake_A_temp, fake_B_temp, cyc_A_temp, cyc_B_temp = exe.run(
                g_A_trainer.infer_program,
                fetch_list=[
                    g_A_trainer.fake_A, g_A_trainer.fake_B, g_A_trainer.cyc_A,
                    g_A_trainer.cyc_B
                ],
                feed={"input_A": tensor_A,
                      "input_B": tensor_B})
            fake_A_temp = np.squeeze(fake_A_temp[0]).transpose([1, 2, 0])
            fake_B_temp = np.squeeze(fake_B_temp[0]).transpose([1, 2, 0])
            cyc_A_temp = np.squeeze(cyc_A_temp[0]).transpose([1, 2, 0])
            cyc_B_temp = np.squeeze(cyc_B_temp[0]).transpose([1, 2, 0])
            input_A_temp = np.squeeze(data_A[0]).transpose([1, 2, 0])
            input_B_temp = np.squeeze(data_B[0]).transpose([1, 2, 0])

            imsave(out_path + "/fakeB_" + str(epoch) + "_" + A_name, (
                (fake_B_temp + 1) * 127.5).astype(np.uint8))
            imsave(out_path + "/fakeA_" + str(epoch) + "_" + B_name, (
                (fake_A_temp + 1) * 127.5).astype(np.uint8))
            imsave(out_path + "/cycA_" + str(epoch) + "_" + A_name, (
                (cyc_A_temp + 1) * 127.5).astype(np.uint8))
            imsave(out_path + "/cycB_" + str(epoch) + "_" + B_name, (
                (cyc_B_temp + 1) * 127.5).astype(np.uint8))
            imsave(out_path + "/inputA_" + str(epoch) + "_" + A_name, (
                (input_A_temp + 1) * 127.5).astype(np.uint8))
            imsave(out_path + "/inputB_" + str(epoch) + "_" + B_name, (
                (input_B_temp + 1) * 127.5).astype(np.uint8))
            i += 1

    def checkpoints(epoch):
        out_path = args.output + "/checkpoints/" + str(epoch)
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        fluid.io.save_persistables(
            exe, out_path + "/g_a", main_program=g_A_trainer.program)
        fluid.io.save_persistables(
            exe, out_path + "/g_b", main_program=g_B_trainer.program)
        fluid.io.save_persistables(
            exe, out_path + "/d_a", main_program=d_A_trainer.program)
        fluid.io.save_persistables(
            exe, out_path + "/d_b", main_program=d_B_trainer.program)
        print("saved checkpoint to {}".format(out_path))
        sys.stdout.flush()

    def init_model():
        assert os.path.exists(
            args.init_model), "[%s] cann't be found." % args.init_mode
        fluid.io.load_persistables(
            exe, args.init_model + "/g_a", main_program=g_A_trainer.program)
        fluid.io.load_persistables(
            exe, args.init_model + "/g_b", main_program=g_B_trainer.program)
        fluid.io.load_persistables(
            exe, args.init_model + "/d_a", main_program=d_A_trainer.program)
        fluid.io.load_persistables(
            exe, args.init_model + "/d_b", main_program=d_B_trainer.program)
        print("Load model from {}".format(args.init_model))

    if args.init_model:
        init_model()
    losses = [[], []]
    t_time = 0
    build_strategy = fluid.BuildStrategy()
    build_strategy.enable_inplace = False
    build_strategy.memory_optimize = False

    exec_strategy = fluid.ExecutionStrategy()
    exec_strategy.num_threads = 1
    exec_strategy.use_experimental_executor = True

    g_A_trainer_program = fluid.CompiledProgram(
        g_A_trainer.program).with_data_parallel(
            loss_name=g_A_trainer.g_loss_A.name,
            build_strategy=build_strategy,
            exec_strategy=exec_strategy)
    g_B_trainer_program = fluid.CompiledProgram(
        g_B_trainer.program).with_data_parallel(
            loss_name=g_B_trainer.g_loss_B.name,
            build_strategy=build_strategy,
            exec_strategy=exec_strategy)
    d_B_trainer_program = fluid.CompiledProgram(
        d_B_trainer.program).with_data_parallel(
            loss_name=d_B_trainer.d_loss_B.name,
            build_strategy=build_strategy,
            exec_strategy=exec_strategy)
    d_A_trainer_program = fluid.CompiledProgram(
        d_A_trainer.program).with_data_parallel(
            loss_name=d_A_trainer.d_loss_A.name,
            build_strategy=build_strategy,
            exec_strategy=exec_strategy)
    for epoch in range(args.epoch):
        batch_id = 0
        for i in range(max_images_num):
            data_A = next(A_reader)
            data_B = next(B_reader)
            tensor_A = fluid.LoDTensor()
            tensor_B = fluid.LoDTensor()
            tensor_A.set(data_A, place)
            tensor_B.set(data_B, place)
            s_time = time.time()
            # optimize the g_A network
            g_A_loss, fake_B_tmp = exe.run(
                g_A_trainer_program,
                fetch_list=[g_A_trainer.g_loss_A, g_A_trainer.fake_B],
                feed={"input_A": tensor_A,
                      "input_B": tensor_B})

            fake_pool_B = B_pool.pool_image(fake_B_tmp)

            # optimize the d_B network
            d_B_loss = exe.run(
                d_B_trainer_program,
                fetch_list=[d_B_trainer.d_loss_B],
                feed={"input_B": tensor_B,
                      "fake_pool_B": fake_pool_B})[0]

            # optimize the g_B network
            g_B_loss, fake_A_tmp = exe.run(
                g_B_trainer_program,
                fetch_list=[g_B_trainer.g_loss_B, g_B_trainer.fake_A],
                feed={"input_A": tensor_A,
                      "input_B": tensor_B})

            fake_pool_A = A_pool.pool_image(fake_A_tmp)

            # optimize the d_A network
            d_A_loss = exe.run(
                d_A_trainer_program,
                fetch_list=[d_A_trainer.d_loss_A],
                feed={"input_A": tensor_A,
                      "fake_pool_A": fake_pool_A})[0]
            batch_time = time.time() - s_time
            t_time += batch_time
            print(
                "epoch{}; batch{}; g_A_loss: {}; d_B_loss: {}; g_B_loss: {}; d_A_loss: {}; "
                "Batch_time_cost: {}".format(epoch, batch_id, g_A_loss[
                    0], d_B_loss[0], g_B_loss[0], d_A_loss[0], batch_time))
            losses[0].append(g_A_loss[0])
            losses[1].append(d_A_loss[0])
            sys.stdout.flush()
            batch_id += 1

        if args.run_test and not args.run_ce:
            test(epoch)
        if args.save_checkpoints and not args.run_ce:
            checkpoints(epoch)
    if args.run_ce:
        print("kpis,g_train_cost,{}".format(np.mean(losses[0])))
        print("kpis,d_train_cost,{}".format(np.mean(losses[1])))
        print("kpis,duration,{}".format(t_time / args.epoch))


if __name__ == "__main__":
    args = parser.parse_args()
    print_arguments(args)
    if args.profile:
        if args.use_gpu:
            with profiler.cuda_profiler("cuda_profiler.txt", 'csv') as nvprof:
                train(args)
        else:
            with profiler.profiler("CPU", sorted_key='total') as cpuprof:
                train(args)
    else:
        train(args)
