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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import random
import sys
import paddle
import argparse
import functools
import time
import numpy as np
from scipy.misc import imsave
import paddle.fluid as fluid
import data_reader
from utility import add_arguments, print_arguments, ImagePool
from trainer import *
from paddle.fluid.dygraph.base import to_variable
import six
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--ce", action="store_true", help="run ce")
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('batch_size',        int,   1,          "Minibatch size.")
add_arg('epoch',             int,   200,        "The number of epoched to be trained.")
add_arg('output',            str,   "./output_0", "The directory the model and the test result to be saved to.")
add_arg('init_model',        str,   None,       "The init model file of directory.")
add_arg('save_checkpoints',  bool,  True,       "Whether to save checkpoints.")
# yapf: enable

lambda_A = 10.0
lambda_B = 10.0
lambda_identity = 0.5
step_per_epoch = 2974


def optimizer_setting(parameters):
    lr = 0.0002
    optimizer = fluid.optimizer.Adam(
        learning_rate=fluid.layers.piecewise_decay(
            boundaries=[
                100 * step_per_epoch, 120 * step_per_epoch,
                140 * step_per_epoch, 160 * step_per_epoch, 180 * step_per_epoch
            ],
            values=[lr, lr * 0.8, lr * 0.6, lr * 0.4, lr * 0.2, lr * 0.1]),
        parameter_list=parameters,
        beta1=0.5)
    return optimizer


def train(args):
    with fluid.dygraph.guard():
        max_images_num = data_reader.max_images_num()
        shuffle = True
        data_shape = [-1] + data_reader.image_shape()
        print(data_shape)
        if args.ce:
            print("ce mode")
            seed = 33
            random.seed(seed)
            np.random.seed(seed)
            fluid.default_startup_program().random_seed = seed
            fluid.default_main_program().random_seed = seed
            shuffle = False

        A_pool = ImagePool()
        B_pool = ImagePool()
        A_reader = paddle.batch(
            data_reader.a_reader(shuffle=shuffle), args.batch_size)()
        B_reader = paddle.batch(
            data_reader.b_reader(shuffle=shuffle), args.batch_size)()
        A_test_reader = data_reader.a_test_reader()
        B_test_reader = data_reader.b_test_reader()

        cycle_gan = Cycle_Gan(input_channel=data_shape[1], istrain=True)

        losses = [[], []]
        t_time = 0

        vars_G = cycle_gan.build_generator_resnet_9blocks_a.parameters(
        ) + cycle_gan.build_generator_resnet_9blocks_b.parameters()
        vars_da = cycle_gan.build_gen_discriminator_a.parameters()
        vars_db = cycle_gan.build_gen_discriminator_b.parameters()

        optimizer1 = optimizer_setting(vars_G)
        optimizer2 = optimizer_setting(vars_da)
        optimizer3 = optimizer_setting(vars_db)

        for epoch in range(args.epoch):
            batch_id = 0
            for i in range(max_images_num):

                data_A = next(A_reader)
                data_B = next(B_reader)

                s_time = time.time()
                data_A = np.array(
                    [data_A[0].reshape(3, 256, 256)]).astype("float32")
                data_B = np.array(
                    [data_B[0].reshape(3, 256, 256)]).astype("float32")
                data_A = to_variable(data_A)
                data_B = to_variable(data_B)

                # optimize the g_A network
                fake_A, fake_B, cyc_A, cyc_B, g_A_loss, g_B_loss, idt_loss_A, idt_loss_B, cyc_A_loss, cyc_B_loss, g_loss = cycle_gan(
                    data_A, data_B, True, False, False)

                g_loss_out = g_loss.numpy()

                g_loss.backward()

                optimizer1.minimize(g_loss)
                cycle_gan.clear_gradients()

                fake_pool_B = B_pool.pool_image(fake_B).numpy()
                fake_pool_B = np.array(
                    [fake_pool_B[0].reshape(3, 256, 256)]).astype("float32")
                fake_pool_B = to_variable(fake_pool_B)

                fake_pool_A = A_pool.pool_image(fake_A).numpy()
                fake_pool_A = np.array(
                    [fake_pool_A[0].reshape(3, 256, 256)]).astype("float32")
                fake_pool_A = to_variable(fake_pool_A)

                # optimize the d_A network
                rec_B, fake_pool_rec_B = cycle_gan(data_B, fake_pool_B, False,
                                                   True, False)
                d_loss_A = (fluid.layers.square(fake_pool_rec_B) +
                            fluid.layers.square(rec_B - 1)) / 2.0
                d_loss_A = fluid.layers.reduce_mean(d_loss_A)

                d_loss_A.backward()
                optimizer2.minimize(d_loss_A)
                cycle_gan.clear_gradients()

                # optimize the d_B network

                rec_A, fake_pool_rec_A = cycle_gan(data_A, fake_pool_A, False,
                                                   False, True)
                d_loss_B = (fluid.layers.square(fake_pool_rec_A) +
                            fluid.layers.square(rec_A - 1)) / 2.0
                d_loss_B = fluid.layers.reduce_mean(d_loss_B)

                d_loss_B.backward()
                optimizer3.minimize(d_loss_B)

                cycle_gan.clear_gradients()

                batch_time = time.time() - s_time
                t_time += batch_time
                print(
                    "epoch{}; batch{}; g_loss:{}; d_A_loss: {}; d_B_loss:{} ; \n g_A_loss: {}; g_A_cyc_loss: {}; g_A_idt_loss: {}; g_B_loss: {}; g_B_cyc_loss:  {}; g_B_idt_loss: {};Batch_time_cost: {}".
                    format(epoch, batch_id, g_loss_out[0],
                           d_loss_A.numpy()[0],
                           d_loss_B.numpy()[0],
                           g_A_loss.numpy()[0],
                           cyc_A_loss.numpy()[0],
                           idt_loss_A.numpy()[0],
                           g_B_loss.numpy()[0],
                           cyc_B_loss.numpy()[0],
                           idt_loss_B.numpy()[0], batch_time))
                with open('logging_train.txt', 'a') as log_file:
                    now = time.strftime("%c")
                    log_file.write(
                    "time: {}; epoch{}; batch{}; d_A_loss: {}; g_A_loss: {}; \
                    g_A_cyc_loss: {}; g_A_idt_loss: {}; d_B_loss: {}; \
                    g_B_loss: {}; g_B_cyc_loss: {}; g_B_idt_loss: {}; \
                    Batch_time_cost: {}\n"
                                          .format(now, epoch, \
                        batch_id, d_loss_A[0], g_A_loss[ 0], cyc_A_loss[0], \
                        idt_loss_A[0], d_loss_B[0], g_A_loss[0], \
                        cyc_B_loss[0], idt_loss_B[0], batch_time))
                losses[0].append(g_A_loss[0])
                losses[1].append(d_loss_A[0])
                sys.stdout.flush()
                batch_id += 1
                if args.ce and batch_id == 500:
                    print("kpis\tg_loss\t%0.3f" % g_loss_out[0])
                    print("kpis\tg_A_loss\t%0.3f" % g_A_loss.numpy()[0])
                    print("kpis\tg_B_loss\t%0.3f" % g_B_loss.numpy()[0])
                    print("kpis\td_A_loss\t%0.3f" % d_loss_A.numpy()[0])
                    print("kpis\td_B_loss\t%0.3f" % d_loss_B.numpy()[0])
                    break

            if args.save_checkpoints:
                fluid.save_dygraph(
                    cycle_gan.state_dict(),
                    args.output + "/checkpoints/{}".format(epoch))


if __name__ == "__main__":
    args = parser.parse_args()
    print_arguments(args)
    train(args)
