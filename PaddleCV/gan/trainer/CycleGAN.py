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
from network.CycleGAN_network import CycleGAN_model
from util import utility
import paddle.fluid as fluid
from paddle.fluid import profiler
import paddle
import sys
import time

lambda_A = 10.0
lambda_B = 10.0
lambda_identity = 0.5


class GTrainer():
    def __init__(self, input_A, input_B, cfg, step_per_epoch):
        self.program = fluid.default_main_program().clone()
        with fluid.program_guard(self.program):
            model = CycleGAN_model()
            self.fake_B = model.network_G(input_A, name="GA", cfg=cfg)
            self.fake_A = model.network_G(input_B, name="GB", cfg=cfg)
            self.cyc_A = model.network_G(self.fake_B, name="GB", cfg=cfg)
            self.cyc_B = model.network_G(self.fake_A, name="GA", cfg=cfg)

            self.infer_program = self.program.clone()
            # Cycle Loss
            diff_A = fluid.layers.abs(
                fluid.layers.elementwise_sub(
                    x=input_A, y=self.cyc_A))
            diff_B = fluid.layers.abs(
                fluid.layers.elementwise_sub(
                    x=input_B, y=self.cyc_B))
            self.cyc_A_loss = fluid.layers.reduce_mean(diff_A) * lambda_A
            self.cyc_B_loss = fluid.layers.reduce_mean(diff_B) * lambda_B
            self.cyc_loss = self.cyc_A_loss + self.cyc_B_loss
            # GAN Loss D_A(G_A(A))
            self.fake_rec_A = model.network_D(self.fake_B, name="DA", cfg=cfg)
            self.G_A = fluid.layers.reduce_mean(
                fluid.layers.square(self.fake_rec_A - 1))
            # GAN Loss D_B(G_B(B))
            self.fake_rec_B = model.network_D(self.fake_A, name="DB", cfg=cfg)
            self.G_B = fluid.layers.reduce_mean(
                fluid.layers.square(self.fake_rec_B - 1))
            self.G = self.G_A + self.G_B
            # Identity Loss G_A
            self.idt_A = model.network_G(input_B, name="GA", cfg=cfg)
            self.idt_loss_A = fluid.layers.reduce_mean(
                fluid.layers.abs(
                    fluid.layers.elementwise_sub(
                        x=input_B, y=self.idt_A))) * lambda_B * lambda_identity
            # Identity Loss G_B
            self.idt_B = model.network_G(input_A, name="GB", cfg=cfg)
            self.idt_loss_B = fluid.layers.reduce_mean(
                fluid.layers.abs(
                    fluid.layers.elementwise_sub(
                        x=input_A, y=self.idt_B))) * lambda_A * lambda_identity

            self.idt_loss = fluid.layers.elementwise_add(self.idt_loss_A,
                                                         self.idt_loss_B)
            self.g_loss = self.cyc_loss + self.G + self.idt_loss

            vars = []
            for var in self.program.list_vars():
                if fluid.io.is_parameter(var) and (var.name.startswith("GA") or
                                                   var.name.startswith("GB")):
                    vars.append(var.name)
            self.param = vars
            lr = cfg.learning_rate
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


class DATrainer():
    def __init__(self, input_B, fake_pool_B, cfg, step_per_epoch):
        self.program = fluid.default_main_program().clone()
        with fluid.program_guard(self.program):
            model = CycleGAN_model()
            self.rec_B = model.network_D(input_B, name="DA", cfg=cfg)
            self.fake_pool_rec_B = model.network_D(
                fake_pool_B, name="DA", cfg=cfg)
            self.d_loss_A = (fluid.layers.square(self.fake_pool_rec_B) +
                             fluid.layers.square(self.rec_B - 1)) / 2.0
            self.d_loss_A = fluid.layers.reduce_mean(self.d_loss_A)

            vars = []
            for var in self.program.list_vars():
                if fluid.io.is_parameter(var) and var.name.startswith("DA"):
                    vars.append(var.name)

            self.param = vars
            lr = cfg.learning_rate
            if cfg.epoch <= 100:
                optimizer = fluid.optimizer.Adam(
                    learning_rate=lr, beta1=0.5, beta2=0.999, name="net_DA")
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
                    name="net_DA")

            optimizer.minimize(self.d_loss_A, parameter_list=vars)


class DBTrainer():
    def __init__(self, input_A, fake_pool_A, cfg, step_per_epoch):
        self.program = fluid.default_main_program().clone()
        with fluid.program_guard(self.program):
            model = CycleGAN_model()
            self.rec_A = model.network_D(input_A, name="DB", cfg=cfg)
            self.fake_pool_rec_A = model.network_D(
                fake_pool_A, name="DB", cfg=cfg)
            self.d_loss_B = (fluid.layers.square(self.fake_pool_rec_A) +
                             fluid.layers.square(self.rec_A - 1)) / 2.0
            self.d_loss_B = fluid.layers.reduce_mean(self.d_loss_B)
            vars = []
            for var in self.program.list_vars():
                if fluid.io.is_parameter(var) and var.name.startswith("DB"):
                    vars.append(var.name)
            self.param = vars
            lr = 0.0002
            if cfg.epoch <= 100:
                optimizer = fluid.optimizer.Adam(
                    learning_rate=lr, beta1=0.5, beta2=0.999, name="net_DA")
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
                    name="net_DB")
            optimizer.minimize(self.d_loss_B, parameter_list=vars)


class CycleGAN(object):
    def add_special_args(self, parser):
        parser.add_argument(
            '--net_G',
            type=str,
            default="resnet_9block",
            help="Choose the CycleGAN generator's network, choose in [resnet_9block|resnet_6block|unet_128|unet_256]"
        )
        parser.add_argument(
            '--net_D',
            type=str,
            default="basic",
            help="Choose the CycleGAN discriminator's network, choose in [basic|nlayers|pixel]"
        )
        parser.add_argument(
            '--d_nlayers',
            type=int,
            default=3,
            help="only used when CycleGAN discriminator is nlayers")
        parser.add_argument(
            '--enable_ce',
            action='store_true',
            help="if set, run the tasks with continuous evaluation logs")
        return parser

    def __init__(self,
                 cfg=None,
                 A_reader=None,
                 B_reader=None,
                 A_test_reader=None,
                 B_test_reader=None,
                 batch_num=1,
                 A_id2name=None,
                 B_id2name=None):
        self.cfg = cfg
        self.A_reader = A_reader
        self.B_reader = B_reader
        self.A_test_reader = A_test_reader
        self.B_test_reader = B_test_reader
        self.batch_num = batch_num
        self.A_id2name = A_id2name
        self.B_id2name = B_id2name

    def build_model(self):
        data_shape = [None, 3, self.cfg.crop_size, self.cfg.crop_size]

        input_A = fluid.data(name='input_A', shape=data_shape, dtype='float32')
        input_B = fluid.data(name='input_B', shape=data_shape, dtype='float32')
        fake_pool_A = fluid.data(
            name='fake_pool_A', shape=data_shape, dtype='float32')
        fake_pool_B = fluid.data(
            name='fake_pool_B', shape=data_shape, dtype='float32')
        # used for continuous evaluation
        if self.cfg.enable_ce:
            fluid.default_startup_program().random_seed = 90

        A_loader = fluid.io.DataLoader.from_generator(
            feed_list=[input_A],
            capacity=4,
            iterable=True,
            use_double_buffer=True)

        B_loader = fluid.io.DataLoader.from_generator(
            feed_list=[input_B],
            capacity=4,
            iterable=True,
            use_double_buffer=True)

        gen_trainer = GTrainer(input_A, input_B, self.cfg, self.batch_num)
        d_A_trainer = DATrainer(input_B, fake_pool_B, self.cfg, self.batch_num)
        d_B_trainer = DBTrainer(input_A, fake_pool_A, self.cfg, self.batch_num)

        # prepare environment
        place = fluid.CUDAPlace(0) if self.cfg.use_gpu else fluid.CPUPlace()

        A_loader.set_batch_generator(
            self.A_reader,
            places=fluid.cuda_places()
            if self.cfg.use_gpu else fluid.cpu_places())
        B_loader.set_batch_generator(
            self.B_reader,
            places=fluid.cuda_places()
            if self.cfg.use_gpu else fluid.cpu_places())

        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())

        A_pool = utility.ImagePool()
        B_pool = utility.ImagePool()

        if self.cfg.init_model:
            utility.init_checkpoints(self.cfg, gen_trainer, "net_G")
            utility.init_checkpoints(self.cfg, d_A_trainer, "net_DA")
            utility.init_checkpoints(self.cfg, d_B_trainer, "net_DB")

        ### memory optim
        build_strategy = fluid.BuildStrategy()
        build_strategy.enable_inplace = True

        gen_trainer_program = fluid.CompiledProgram(
            gen_trainer.program).with_data_parallel(
                loss_name=gen_trainer.g_loss.name,
                build_strategy=build_strategy)
        d_A_trainer_program = fluid.CompiledProgram(
            d_A_trainer.program).with_data_parallel(
                loss_name=d_A_trainer.d_loss_A.name,
                build_strategy=build_strategy)
        d_B_trainer_program = fluid.CompiledProgram(
            d_B_trainer.program).with_data_parallel(
                loss_name=d_B_trainer.d_loss_B.name,
                build_strategy=build_strategy)

        t_time = 0

        total_train_batch = 0  # NOTE :used for benchmark
        for epoch_id in range(self.cfg.epoch):
            batch_id = 0
            for data_A, data_B in zip(A_loader(), B_loader()):
                if self.cfg.max_iter and total_train_batch == self.cfg.max_iter: # used for benchmark
                    return
                s_time = time.time()
                tensor_A, tensor_B = data_A[0]['input_A'], data_B[0]['input_B']
                ## optimize the g_A network
                g_A_loss, g_A_cyc_loss, g_A_idt_loss, g_B_loss, g_B_cyc_loss,\
                g_B_idt_loss, fake_A_tmp, fake_B_tmp = exe.run(
                    gen_trainer_program,
                    fetch_list=[
                        gen_trainer.G_A, gen_trainer.cyc_A_loss,
                        gen_trainer.idt_loss_A, gen_trainer.G_B,
                        gen_trainer.cyc_B_loss, gen_trainer.idt_loss_B,
                        gen_trainer.fake_A, gen_trainer.fake_B
                    ],
                    feed={"input_A": tensor_A,
                          "input_B": tensor_B})

                fake_pool_B = B_pool.pool_image(fake_B_tmp)
                fake_pool_A = A_pool.pool_image(fake_A_tmp)

                if self.cfg.enable_ce:
                    fake_pool_B = fake_B_tmp
                    fake_pool_A = fake_A_tmp

                # optimize the d_A network
                d_A_loss = exe.run(
                    d_A_trainer_program,
                    fetch_list=[d_A_trainer.d_loss_A],
                    feed={"input_B": tensor_B,
                          "fake_pool_B": fake_pool_B})[0]

                # optimize the d_B network
                d_B_loss = exe.run(
                    d_B_trainer_program,
                    fetch_list=[d_B_trainer.d_loss_B],
                    feed={"input_A": tensor_A,
                          "fake_pool_A": fake_pool_A})[0]

                batch_time = time.time() - s_time
                t_time += batch_time
                if batch_id % self.cfg.print_freq == 0:
                    print("epoch{}: batch{}: \n\
                         d_A_loss: {}; g_A_loss: {}; g_A_cyc_loss: {}; g_A_idt_loss: {}; \n\
                         d_B_loss: {}; g_B_loss: {}; g_B_cyc_loss: {}; g_B_idt_loss: {}; \n\
                         Batch_time_cost: {}".format(
                        epoch_id, batch_id, d_A_loss[0], g_A_loss[0],
                        g_A_cyc_loss[0], g_A_idt_loss[0], d_B_loss[0], g_B_loss[
                            0], g_B_cyc_loss[0], g_B_idt_loss[0], batch_time))

                sys.stdout.flush()
                batch_id += 1
                #NOTE: used for benchmark
                total_train_batch += 1  # used for benchmark
                # profiler tools
                if self.cfg.profile and epoch_id == 0 and batch_id == self.cfg.print_freq:
                    profiler.reset_profiler()
                elif self.cfg.profile and epoch_id == 0 and batch_id == self.cfg.print_freq + 5:
                    return
                # used for continuous evaluation
                if self.cfg.enable_ce and batch_id == 10:
                    break

            if self.cfg.run_test:
                A_image_name = fluid.data(
                    name='A_image_name', shape=[None, 1], dtype='int32')
                B_image_name = fluid.data(
                    name='B_image_name', shape=[None, 1], dtype='int32')
                A_test_loader = fluid.io.DataLoader.from_generator(
                    feed_list=[input_A, A_image_name],
                    capacity=4,
                    iterable=True,
                    use_double_buffer=True)

                B_test_loader = fluid.io.DataLoader.from_generator(
                    feed_list=[input_B, B_image_name],
                    capacity=4,
                    iterable=True,
                    use_double_buffer=True)

                A_test_loader.set_batch_generator(
                    self.A_test_reader,
                    places=fluid.cuda_places()
                    if self.cfg.use_gpu else fluid.cpu_places())
                B_test_loader.set_batch_generator(
                    self.B_test_reader,
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
                    A_test_loader,
                    B_test_loader,
                    A_id2name=self.A_id2name,
                    B_id2name=self.B_id2name)

            if self.cfg.save_checkpoints:
                utility.checkpoints(epoch_id, self.cfg, gen_trainer,
                                    "net_G")
                utility.checkpoints(epoch_id, self.cfg, d_A_trainer,
                                    "net_DA")
                utility.checkpoints(epoch_id, self.cfg, d_B_trainer,
                                    "net_DB")

        # used for continuous evaluation
        if self.cfg.enable_ce:
            device_num = fluid.core.get_cuda_device_count(
            ) if self.cfg.use_gpu else 1
            print("kpis\tcyclegan_g_A_loss_card{}\t{}".format(device_num,
                                                              g_A_loss[0]))
            print("kpis\tcyclegan_g_A_cyc_loss_card{}\t{}".format(
                device_num, g_A_cyc_loss[0]))
            print("kpis\tcyclegan_g_A_idt_loss_card{}\t{}".format(
                device_num, g_A_idt_loss[0]))
            print("kpis\tcyclegan_d_A_loss_card{}\t{}".format(device_num,
                                                              d_A_loss[0]))
            print("kpis\tcyclegan_g_B_loss_card{}\t{}".format(device_num,
                                                              g_B_loss[0]))
            print("kpis\tcyclegan_g_B_cyc_loss_card{}\t{}".format(
                device_num, g_B_cyc_loss[0]))
            print("kpis\tcyclegan_g_B_idt_loss_card{}\t{}".format(
                device_num, g_B_idt_loss[0]))
            print("kpis\tcyclegan_d_B_loss_card{}\t{}".format(device_num,
                                                              d_B_loss[0]))
            print("kpis\tcyclegan_Batch_time_cost_card{}\t{}".format(
                device_num, batch_time))
