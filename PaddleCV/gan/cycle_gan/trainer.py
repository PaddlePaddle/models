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
from model import build_generator_resnet_9blocks, build_gen_discriminator
import paddle.fluid as fluid

step_per_epoch = 1335
cycle_loss_factor = 10.0


class GATrainer():
    def __init__(self, input_A, input_B):
        self.program = fluid.default_main_program().clone()
        with fluid.program_guard(self.program):
            self.fake_B = build_generator_resnet_9blocks(input_A, name="g_A")
            self.fake_A = build_generator_resnet_9blocks(input_B, name="g_B")
            self.cyc_A = build_generator_resnet_9blocks(self.fake_B, "g_B")
            self.cyc_B = build_generator_resnet_9blocks(self.fake_A, "g_A")
            self.infer_program = self.program.clone()
            diff_A = fluid.layers.abs(
                fluid.layers.elementwise_sub(
                    x=input_A, y=self.cyc_A))
            diff_B = fluid.layers.abs(
                fluid.layers.elementwise_sub(
                    x=input_B, y=self.cyc_B))
            self.cyc_loss = (
                fluid.layers.reduce_mean(diff_A) +
                fluid.layers.reduce_mean(diff_B)) * cycle_loss_factor
            self.fake_rec_B = build_gen_discriminator(self.fake_B, "d_B")
            self.disc_loss_B = fluid.layers.reduce_mean(
                fluid.layers.square(self.fake_rec_B - 1))
            self.g_loss_A = fluid.layers.elementwise_add(self.cyc_loss,
                                                         self.disc_loss_B)
            vars = []
            for var in self.program.list_vars():
                if fluid.io.is_parameter(var) and var.name.startswith("g_A"):
                    vars.append(var.name)
            self.param = vars
            lr = 0.0002
            optimizer = fluid.optimizer.Adam(
                learning_rate=fluid.layers.piecewise_decay(
                    boundaries=[
                        100 * step_per_epoch, 120 * step_per_epoch,
                        140 * step_per_epoch, 160 * step_per_epoch,
                        180 * step_per_epoch
                    ],
                    values=[
                        lr, lr * 0.8, lr * 0.6, lr * 0.4, lr * 0.2, lr * 0.1
                    ]),
                beta1=0.5,
                name="g_A")
            optimizer.minimize(self.g_loss_A, parameter_list=vars)


class GBTrainer():
    def __init__(self, input_A, input_B):
        self.program = fluid.default_main_program().clone()
        with fluid.program_guard(self.program):
            self.fake_B = build_generator_resnet_9blocks(input_A, name="g_A")
            self.fake_A = build_generator_resnet_9blocks(input_B, name="g_B")
            self.cyc_A = build_generator_resnet_9blocks(self.fake_B, "g_B")
            self.cyc_B = build_generator_resnet_9blocks(self.fake_A, "g_A")
            self.infer_program = self.program.clone()
            diff_A = fluid.layers.abs(
                fluid.layers.elementwise_sub(
                    x=input_A, y=self.cyc_A))
            diff_B = fluid.layers.abs(
                fluid.layers.elementwise_sub(
                    x=input_B, y=self.cyc_B))
            self.cyc_loss = (
                fluid.layers.reduce_mean(diff_A) +
                fluid.layers.reduce_mean(diff_B)) * cycle_loss_factor
            self.fake_rec_A = build_gen_discriminator(self.fake_A, "d_A")
            disc_loss_A = fluid.layers.reduce_mean(
                fluid.layers.square(self.fake_rec_A - 1))
            self.g_loss_B = fluid.layers.elementwise_add(self.cyc_loss,
                                                         disc_loss_A)
            vars = []
            for var in self.program.list_vars():
                if fluid.io.is_parameter(var) and var.name.startswith("g_B"):
                    vars.append(var.name)
            self.param = vars
            lr = 0.0002
            optimizer = fluid.optimizer.Adam(
                learning_rate=fluid.layers.piecewise_decay(
                    boundaries=[
                        100 * step_per_epoch, 120 * step_per_epoch,
                        140 * step_per_epoch, 160 * step_per_epoch,
                        180 * step_per_epoch
                    ],
                    values=[
                        lr, lr * 0.8, lr * 0.6, lr * 0.4, lr * 0.2, lr * 0.1
                    ]),
                beta1=0.5,
                name="g_B")
            optimizer.minimize(self.g_loss_B, parameter_list=vars)


class DATrainer():
    def __init__(self, input_A, fake_pool_A):
        self.program = fluid.default_main_program().clone()
        with fluid.program_guard(self.program):
            self.rec_A = build_gen_discriminator(input_A, "d_A")
            self.fake_pool_rec_A = build_gen_discriminator(fake_pool_A, "d_A")
            self.d_loss_A = (fluid.layers.square(self.fake_pool_rec_A) +
                             fluid.layers.square(self.rec_A - 1)) / 2.0
            self.d_loss_A = fluid.layers.reduce_mean(self.d_loss_A)

            optimizer = fluid.optimizer.Adam(learning_rate=0.0002, beta1=0.5)
            optimizer._name = "d_A"
            vars = []
            for var in self.program.list_vars():
                if fluid.io.is_parameter(var) and var.name.startswith("d_A"):
                    vars.append(var.name)

            self.param = vars
            lr = 0.0002
            optimizer = fluid.optimizer.Adam(
                learning_rate=fluid.layers.piecewise_decay(
                    boundaries=[
                        100 * step_per_epoch, 120 * step_per_epoch,
                        140 * step_per_epoch, 160 * step_per_epoch,
                        180 * step_per_epoch
                    ],
                    values=[
                        lr, lr * 0.8, lr * 0.6, lr * 0.4, lr * 0.2, lr * 0.1
                    ]),
                beta1=0.5,
                name="d_A")

            optimizer.minimize(self.d_loss_A, parameter_list=vars)


class DBTrainer():
    def __init__(self, input_B, fake_pool_B):
        self.program = fluid.default_main_program().clone()
        with fluid.program_guard(self.program):
            self.rec_B = build_gen_discriminator(input_B, "d_B")
            self.fake_pool_rec_B = build_gen_discriminator(fake_pool_B, "d_B")
            self.d_loss_B = (fluid.layers.square(self.fake_pool_rec_B) +
                             fluid.layers.square(self.rec_B - 1)) / 2.0
            self.d_loss_B = fluid.layers.reduce_mean(self.d_loss_B)
            optimizer = fluid.optimizer.Adam(learning_rate=0.0002, beta1=0.5)
            vars = []
            for var in self.program.list_vars():
                if fluid.io.is_parameter(var) and var.name.startswith("d_B"):
                    vars.append(var.name)
            self.param = vars
            lr = 0.0002
            optimizer = fluid.optimizer.Adam(
                learning_rate=fluid.layers.piecewise_decay(
                    boundaries=[
                        100 * step_per_epoch, 120 * step_per_epoch,
                        140 * step_per_epoch, 160 * step_per_epoch,
                        180 * step_per_epoch
                    ],
                    values=[
                        lr, lr * 0.8, lr * 0.6, lr * 0.4, lr * 0.2, lr * 0.1
                    ]),
                beta1=0.5,
                name="d_B")
            optimizer.minimize(self.d_loss_B, parameter_list=vars)
