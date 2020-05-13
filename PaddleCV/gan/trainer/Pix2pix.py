#copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from network.Pix2pix_network import Pix2pix_model
from util import utility
import paddle.fluid as fluid
from paddle.fluid import profiler
import sys
import time
import numpy as np


class GTrainer():
    def __init__(self, input_A, input_B, cfg, step_per_epoch):
        self.program = fluid.default_main_program().clone()
        with fluid.program_guard(self.program):
            model = Pix2pix_model()
            self.fake_B = model.network_G(input_A, "generator", cfg=cfg)
            self.infer_program = self.program.clone()
            AB = fluid.layers.concat([input_A, self.fake_B], 1)
            self.pred = model.network_D(AB, "discriminator", cfg)
            batch = fluid.layers.shape(self.pred)[0]
            if cfg.gan_mode == "lsgan":
                ones = fluid.layers.fill_constant(
                    shape=[batch] + list(self.pred.shape[1:]),
                    value=1,
                    dtype='float32')
                self.g_loss_gan = fluid.layers.reduce_mean(
                    fluid.layers.square(
                        fluid.layers.elementwise_sub(
                            x=self.pred, y=ones)))
            elif cfg.gan_mode == "vanilla":
                pred_shape = self.pred.shape
                self.pred = fluid.layers.reshape(
                    self.pred,
                    [-1, pred_shape[1] * pred_shape[2] * pred_shape[3]],
                    inplace=True)
                ones = fluid.layers.fill_constant(
                    shape=[batch] + list(self.pred.shape[1:]),
                    value=1,
                    dtype='float32')
                self.g_loss_gan = fluid.layers.mean(
                    fluid.layers.sigmoid_cross_entropy_with_logits(
                        x=self.pred, label=ones))
            else:
                raise NotImplementedError("gan_mode {} is not support!".format(
                    cfg.gan_mode))

            self.g_loss_L1 = fluid.layers.reduce_mean(
                fluid.layers.abs(
                    fluid.layers.elementwise_sub(
                        x=input_B, y=self.fake_B))) * cfg.lambda_L1
            self.g_loss = fluid.layers.elementwise_add(self.g_loss_L1,
                                                       self.g_loss_gan)
            lr = cfg.learning_rate
            vars = []
            for var in self.program.list_vars():
                if fluid.io.is_parameter(var) and var.name.startswith(
                        "generator"):
                    vars.append(var.name)
            self.param = vars
            if cfg.epoch <= 100:
                optimizer = fluid.optimizer.Adam(
                    learning_rate=lr, beta1=0.5, beta2=0.999, name="net_G")
            else:
                optimizer = fluid.optimizer.Adam(
                    learning_rate=fluid.layers.piecewise_decay(
                        boundaries=[99 * step_per_epoch] + [
                            x * step_per_epoch
                            for x in range(100, cfg.epoch - 1)
                        ],
                        values=[lr] + [
                            lr * (1.0 - (x - 99.0) / 101.0)
                            for x in range(100, cfg.epoch)
                        ]),
                    beta1=0.5,
                    beta2=0.999,
                    name="net_G")
            optimizer.minimize(self.g_loss, parameter_list=vars)


class DTrainer():
    def __init__(self, input_A, input_B, fake_B, cfg, step_per_epoch):
        self.program = fluid.default_main_program().clone()
        lr = cfg.learning_rate
        with fluid.program_guard(self.program):
            model = Pix2pix_model()
            self.real_AB = fluid.layers.concat([input_A, input_B], 1)
            self.fake_AB = fluid.layers.concat([input_A, fake_B], 1)
            self.pred_real = model.network_D(
                self.real_AB, "discriminator", cfg=cfg)
            self.pred_fake = model.network_D(
                self.fake_AB, "discriminator", cfg=cfg)
            batch = fluid.layers.shape(input_A)[0]
            if cfg.gan_mode == "lsgan":
                ones = fluid.layers.fill_constant(
                    shape=[batch] + list(self.pred_real.shape[1:]),
                    value=1,
                    dtype='float32')
                self.d_loss_real = fluid.layers.reduce_mean(
                    fluid.layers.square(
                        fluid.layers.elementwise_sub(
                            x=self.pred_real, y=ones)))
                self.d_loss_fake = fluid.layers.reduce_mean(
                    fluid.layers.square(x=self.pred_fake))
            elif cfg.gan_mode == "vanilla":
                pred_shape = self.pred_real.shape
                self.pred_real = fluid.layers.reshape(
                    self.pred_real,
                    [-1, pred_shape[1] * pred_shape[2] * pred_shape[3]],
                    inplace=True)
                self.pred_fake = fluid.layers.reshape(
                    self.pred_fake,
                    [-1, pred_shape[1] * pred_shape[2] * pred_shape[3]],
                    inplace=True)
                zeros = fluid.layers.fill_constant(
                    shape=[batch] + list(self.pred_fake.shape[1:]),
                    value=0,
                    dtype='float32')
                ones = fluid.layers.fill_constant(
                    shape=[batch] + list(self.pred_real.shape[1:]),
                    value=1,
                    dtype='float32')
                self.d_loss_real = fluid.layers.mean(
                    fluid.layers.sigmoid_cross_entropy_with_logits(
                        x=self.pred_real, label=ones))
                self.d_loss_fake = fluid.layers.mean(
                    fluid.layers.sigmoid_cross_entropy_with_logits(
                        x=self.pred_fake, label=zeros))
            else:
                raise NotImplementedError("gan_mode {} is not support!".format(
                    cfg.gan_mode))

            self.d_loss = 0.5 * (self.d_loss_real + self.d_loss_fake)
            vars = []
            for var in self.program.list_vars():
                if fluid.io.is_parameter(var) and var.name.startswith(
                        "discriminator"):
                    vars.append(var.name)

            self.param = vars
            if cfg.epoch <= 100:
                optimizer = fluid.optimizer.Adam(
                    learning_rate=lr, beta1=0.5, beta2=0.999, name="net_D")
            else:
                optimizer = fluid.optimizer.Adam(
                    learning_rate=fluid.layers.piecewise_decay(
                        boundaries=[99 * step_per_epoch] + [
                            x * step_per_epoch
                            for x in range(100, cfg.epoch - 1)
                        ],
                        values=[lr] + [
                            lr * (1.0 - (x - 99.0) / 101.0)
                            for x in range(100, cfg.epoch)
                        ]),
                    beta1=0.5,
                    beta2=0.999,
                    name="net_D")

            optimizer.minimize(self.d_loss, parameter_list=vars)


class Pix2pix(object):
    def add_special_args(self, parser):
        parser.add_argument(
            '--net_G',
            type=str,
            default="unet_256",
            help="Choose the Pix2pix generator's network, choose in [resnet_9block|resnet_6block|unet_128|unet_256]"
        )
        parser.add_argument(
            '--net_D',
            type=str,
            default="basic",
            help="Choose the Pix2pix discriminator's network, choose in [basic|nlayers|pixel]"
        )
        parser.add_argument(
            '--d_nlayers',
            type=int,
            default=3,
            help="only used when Pix2pix discriminator is nlayers")
        parser.add_argument(
            '--enable_ce',
            action='store_true',
            help="if set, run the tasks with continuous evaluation logs")
        return parser

    def __init__(self,
                 cfg=None,
                 train_reader=None,
                 test_reader=None,
                 batch_num=1,
                 id2name=None):
        self.cfg = cfg
        self.train_reader = train_reader
        self.test_reader = test_reader
        self.batch_num = batch_num
        self.id2name = id2name

    def build_model(self):
        data_shape = [None, 3, self.cfg.crop_size, self.cfg.crop_size]

        input_A = fluid.data(name='input_A', shape=data_shape, dtype='float32')
        input_B = fluid.data(name='input_B', shape=data_shape, dtype='float32')
        input_fake = fluid.data(
            name='input_fake', shape=data_shape, dtype='float32')
        # used for continuous evaluation        
        if self.cfg.enable_ce:
            fluid.default_startup_program().random_seed = 90

        loader = fluid.io.DataLoader.from_generator(
            feed_list=[input_A, input_B],
            capacity=4,
            iterable=True,
            use_double_buffer=True)

        gen_trainer = GTrainer(input_A, input_B, self.cfg, self.batch_num)
        dis_trainer = DTrainer(input_A, input_B, input_fake, self.cfg,
                               self.batch_num)

        # prepare environment
        place = fluid.CUDAPlace(0) if self.cfg.use_gpu else fluid.CPUPlace()
        loader.set_batch_generator(
            self.train_reader,
            places=fluid.cuda_places()
            if self.cfg.use_gpu else fluid.cpu_places())
        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())

        if self.cfg.init_model:
            utility.init_checkpoints(self.cfg, gen_trainer, "net_G")
            utility.init_checkpoints(self.cfg, dis_trainer, "net_D")

        ### memory optim
        build_strategy = fluid.BuildStrategy()

        gen_trainer_program = fluid.CompiledProgram(
            gen_trainer.program).with_data_parallel(
                loss_name=gen_trainer.g_loss.name,
                build_strategy=build_strategy)
        dis_trainer_program = fluid.CompiledProgram(
            dis_trainer.program).with_data_parallel(
                loss_name=dis_trainer.d_loss.name,
                build_strategy=build_strategy)

        t_time = 0

        total_train_batch = 0  # used for benchmark

        for epoch_id in range(self.cfg.epoch):
            batch_id = 0
            for tensor in loader():
                if self.cfg.max_iter and total_train_batch == self.cfg.max_iter:  # used for benchmark
                    return
                s_time = time.time()

                # optimize the generator network
                g_loss_gan, g_loss_l1, fake_B_tmp = exe.run(
                    gen_trainer_program,
                    fetch_list=[
                        gen_trainer.g_loss_gan, gen_trainer.g_loss_L1,
                        gen_trainer.fake_B
                    ],
                    feed=tensor)

                devices_num = utility.get_device_num(self.cfg)
                fake_per_device = int(len(fake_B_tmp) / devices_num)
                for dev in range(devices_num):
                    tensor[dev]['input_fake'] = fake_B_tmp[
                        dev * fake_per_device:(dev + 1) * fake_per_device]

                # optimize the discriminator network
                d_loss_real, d_loss_fake = exe.run(dis_trainer_program,
                                                   fetch_list=[
                                                       dis_trainer.d_loss_real,
                                                       dis_trainer.d_loss_fake
                                                   ],
                                                   feed=tensor)

                batch_time = time.time() - s_time
                t_time += batch_time
                if batch_id % self.cfg.print_freq == 0:
                    print("epoch{}: batch{}: \n\
                         g_loss_gan: {}; g_loss_l1: {}; \n\
                         d_loss_real: {}; d_loss_fake: {}; \n\
                         Batch_time_cost: {}"
                          .format(epoch_id, batch_id, g_loss_gan[0], g_loss_l1[
                              0], d_loss_real[0], d_loss_fake[0], batch_time))

                sys.stdout.flush()
                batch_id += 1
                total_train_batch += 1  # used for benchmark
                # profiler tools
                if self.cfg.profile and epoch_id == 0 and batch_id == self.cfg.print_freq:
                    profiler.reset_profiler()
                elif self.cfg.profile and epoch_id == 0 and batch_id == self.cfg.print_freq + 5:
                    return

            if self.cfg.run_test:
                image_name = fluid.data(
                    name='image_name',
                    shape=[None, self.cfg.batch_size],
                    dtype="int32")
                test_loader = fluid.io.DataLoader.from_generator(
                    feed_list=[input_A, input_B, image_name],
                    capacity=4,
                    iterable=True,
                    use_double_buffer=True)
                test_loader.set_batch_generator(
                    self.test_reader,
                    places=fluid.cuda_places()
                    if self.cfg.use_gpu else fluid.cpu_places())
                test_program = gen_trainer.infer_program
                utility.save_test_image(
                    epoch_id,
                    self.cfg,
                    exe,
                    place,
                    test_program,
                    gen_trainer,
                    test_loader,
                    A_id2name=self.id2name)

            if self.cfg.save_checkpoints:
                utility.checkpoints(epoch_id, self.cfg, gen_trainer, "net_G")
                utility.checkpoints(epoch_id, self.cfg, dis_trainer, "net_D")
        if self.cfg.enable_ce:
            device_num = fluid.core.get_cuda_device_count(
            ) if self.cfg.use_gpu else 1
            print("kpis\tpix2pix_g_loss_gan_card{}\t{}".format(device_num,
                                                               g_loss_gan[0]))
            print("kpis\tpix2pix_g_loss_l1_card{}\t{}".format(device_num,
                                                              g_loss_l1[0]))
            print("kpis\tpix2pix_d_loss_real_card{}\t{}".format(device_num,
                                                                d_loss_real[0]))
            print("kpis\tpix2pix_d_loss_fake_card{}\t{}".format(device_num,
                                                                d_loss_fake[0]))
            print("kpis\tpix2pix_Batch_time_cost_card{}\t{}".format(device_num,
                                                                    batch_time))
