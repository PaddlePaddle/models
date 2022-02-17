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
from network.SPADE_network import SPADE_model
from util import utility
import paddle.fluid as fluid
import sys
import os
import time
import network.vgg as vgg
import pickle as pkl
import numpy as np


class GTrainer():
    def __init__(self, input_label, input_img, input_ins, cfg, step_per_epoch):
        self.cfg = cfg
        self.program = fluid.default_main_program().clone()
        with fluid.program_guard(self.program):
            model = SPADE_model()
            input = input_label
            if not cfg.no_instance:
                input = fluid.layers.concat([input_label, input_ins], 1)
            self.fake_B = model.network_G(input, "generator", cfg=cfg)
            self.infer_program = self.program.clone()
            fake_concat = fluid.layers.concat([input, self.fake_B], 1)
            real_concat = fluid.layers.concat([input, input_img], 1)
            fake_and_real = fluid.layers.concat([fake_concat, real_concat], 0)
            pred = model.network_D(fake_and_real, "discriminator", cfg)
            if type(pred) == list:
                self.pred_fake = []
                self.pred_real = []
                for p in pred:
                    self.pred_fake.append(
                        [tensor[:tensor.shape[0] // 2] for tensor in p])
                    self.pred_real.append(
                        [tensor[tensor.shape[0] // 2:] for tensor in p])
            else:
                self.pred_fake = pred[:pred.shape[0] // 2]
                self.pred_real = pred[pred.shape[0] // 2:]

            ###GAN Loss hinge
            if isinstance(self.pred_fake, list):
                self.gan_loss = 0
                for pred_i in self.pred_fake:
                    if isinstance(pred_i, list):
                        pred_i = pred_i[-1]
                    loss_i = -1 * fluid.layers.reduce_mean(pred_i)
                    self.gan_loss += loss_i
                self.gan_loss /= len(self.pred_fake)
            else:
                self.gan_loss = -1 * fluid.layers.reduce_mean(self.pred_fake)
            #####GAN Feat loss
            num_D = len(self.pred_fake)
            self.gan_feat_loss = 0.0
            for i in range(num_D):
                num_intermediate_outputs = len(self.pred_fake[i]) - 1
                for j in range(num_intermediate_outputs):
                    self.gan_feat_loss = fluid.layers.reduce_mean(
                        fluid.layers.abs(
                            fluid.layers.elementwise_sub(
                                x=self.pred_fake[i][j], y=self.pred_real[i][
                                    j]))) * cfg.lambda_feat / num_D
            ########VGG Feat loss
            weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
            self.vgg = vgg.VGG19()
            fake_vgg = self.vgg.net(self.fake_B)
            real_vgg = self.vgg.net(input_img)
            self.vgg_loss = 0.0
            for i in range(len(fake_vgg)):
                self.vgg_loss += weights[i] * fluid.layers.reduce_mean(
                    fluid.layers.abs(
                        fluid.layers.elementwise_sub(
                            x=fake_vgg[i], y=real_vgg[i])))
            self.g_loss = (
                self.gan_loss + self.gan_feat_loss + self.vgg_loss) / 3
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
    def __init__(self, input_label, input_img, input_ins, fake_B, cfg,
                 step_per_epoch):
        self.program = fluid.default_main_program().clone()
        lr = cfg.learning_rate
        with fluid.program_guard(self.program):
            model = SPADE_model()
            input = input_label
            if not cfg.no_instance:
                input = fluid.layers.concat([input_label, input_ins], 1)
            fake_concat = fluid.layers.concat([input, fake_B], 1)
            real_concat = fluid.layers.concat([input, input_img], 1)
            fake_and_real = fluid.layers.concat([fake_concat, real_concat], 0)
            pred = model.network_D(fake_and_real, "discriminator", cfg)
            if type(pred) == list:
                self.pred_fake = []
                self.pred_real = []
                for p in pred:
                    self.pred_fake.append(
                        [tensor[:tensor.shape[0] // 2] for tensor in p])
                    self.pred_real.append(
                        [tensor[tensor.shape[0] // 2:] for tensor in p])
            else:
                self.pred_fake = pred[:pred.shape[0] // 2]
                self.pred_real = pred[pred.shape[0] // 2:]

            #####gan loss
            self.gan_loss_fake = 0
            for pred_i in self.pred_fake:
                pred_shape = fluid.layers.shape(pred_i[-1])
                zeros = fluid.layers.fill_constant(
                    shape=pred_shape, value=0, dtype='float32')
                if isinstance(pred_i, list):
                    pred_i = pred_i[-1]
                minval = fluid.layers.elementwise_min(-1 * pred_i - 1, zeros)
                loss_i = -1 * fluid.layers.reduce_mean(minval)
                self.gan_loss_fake += loss_i
            self.gan_loss_fake /= len(self.pred_fake)

            self.gan_loss_real = 0
            for pred_i in self.pred_real:
                pred_shape = fluid.layers.shape(pred_i[-1])
                zeros = fluid.layers.fill_constant(
                    shape=pred_shape, value=0, dtype='float32')
                if isinstance(pred_i, list):
                    pred_i = pred_i[-1]
                minval = fluid.layers.elementwise_min(pred_i - 1, zeros)
                loss_i = -1 * fluid.layers.reduce_mean(minval)
                self.gan_loss_real += loss_i
            self.gan_loss_real /= len(self.pred_real)

            self.d_loss = 0.5 * (self.gan_loss_real + self.gan_loss_fake)
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


class SPADE(object):
    def add_special_args(self, parser):
        parser.add_argument(
            '--vgg19_pretrain',
            type=str,
            default="./VGG19_pretrained",
            help="VGG19 pretrained model for vgg loss")
        parser.add_argument(
            '--crop_width',
            type=int,
            default=1024,
            help="crop width for training SPADE")
        parser.add_argument(
            '--crop_height',
            type=int,
            default=512,
            help="crop height for training SPADE")
        parser.add_argument(
            '--load_width',
            type=int,
            default=1124,
            help="load width for training SPADE")
        parser.add_argument(
            '--load_height',
            type=int,
            default=612,
            help="load height for training SPADE")
        parser.add_argument(
            '--d_nlayers',
            type=int,
            default=4,
            help="num of discriminator layers for SPADE")
        parser.add_argument(
            '--label_nc', type=int, default=36, help="label numbers of SPADE")
        parser.add_argument(
            '--ngf',
            type=int,
            default=64,
            help="base channels of generator in SPADE")
        parser.add_argument(
            '--ndf',
            type=int,
            default=64,
            help="base channels of discriminator in SPADE")
        parser.add_argument(
            '--num_D',
            type=int,
            default=2,
            help="number of discriminators in SPADE")
        parser.add_argument(
            '--lambda_feat',
            type=float,
            default=10,
            help="weight term of feature loss")
        parser.add_argument(
            '--lambda_vgg',
            type=float,
            default=10,
            help="weight term of vgg loss")
        parser.add_argument(
            '--no_instance',
            type=bool,
            default=False,
            help="Whether to use instance label.")
        parser.add_argument(
            '--enable_ce',
            type=bool,
            default=False,
            help="If set True, enable continuous evaluation job.")
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
        data_shape = [None, 3, self.cfg.crop_height, self.cfg.crop_width]
        label_shape = [
            None, self.cfg.label_nc, self.cfg.crop_height, self.cfg.crop_width
        ]
        edge_shape = [None, 1, self.cfg.crop_height, self.cfg.crop_width]

        input_A = fluid.data(
            name='input_label', shape=label_shape, dtype='float32')
        input_B = fluid.data(
            name='input_img', shape=data_shape, dtype='float32')
        input_C = fluid.data(
            name='input_ins', shape=edge_shape, dtype='float32')
        input_fake = fluid.data(
            name='input_fake', shape=data_shape, dtype='float32')
        # used for continuous evaluation
        if self.cfg.enable_ce:
            fluid.default_startup_program().random_seed = 90

        gen_trainer = GTrainer(input_A, input_B, input_C, self.cfg,
                               self.batch_num)
        dis_trainer = DTrainer(input_A, input_B, input_C, input_fake, self.cfg,
                               self.batch_num)
        loader = fluid.io.DataLoader.from_generator(
            feed_list=[input_A, input_B, input_C],
            capacity=4,  ## batch_size * 4
            iterable=True,
            use_double_buffer=True)
        loader.set_batch_generator(
            self.train_reader,
            places=fluid.cuda_places()
            if self.cfg.use_gpu else fluid.cpu_places())

        # prepare environment
        place = fluid.CUDAPlace(0) if self.cfg.use_gpu else fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())

        if not os.path.exists(self.cfg.vgg19_pretrain):
            print(
                "directory VGG19_pretrain NOT EXIST!!! Please download VGG19 first."
            )
            sys.exit(1)
        gen_trainer.vgg.load_vars(exe, gen_trainer.program,
                                  self.cfg.vgg19_pretrain)

        if self.cfg.init_model:
            utility.init_checkpoints(self.cfg, gen_trainer, "net_G")
            utility.init_checkpoints(self.cfg, dis_trainer, "net_D")

        ### memory optim
        build_strategy = fluid.BuildStrategy()
        build_strategy.enable_inplace = True
        build_strategy.sync_batch_norm = False

        gen_trainer_program = fluid.CompiledProgram(
            gen_trainer.program).with_data_parallel(
                loss_name=gen_trainer.g_loss.name,
                build_strategy=build_strategy)
        dis_trainer_program = fluid.CompiledProgram(
            dis_trainer.program).with_data_parallel(
                loss_name=dis_trainer.d_loss.name,
                build_strategy=build_strategy)
        # used for continuous evaluation  
        if self.cfg.enable_ce:
            gen_trainer_program.random_seed = 90
            dis_trainer_program.random_seed = 90

        t_time = 0

        for epoch_id in range(self.cfg.epoch):
            batch_id = 0
            for tensor in loader():
                data_A, data_B, data_C = tensor[0]['input_label'], tensor[0][
                    'input_img'], tensor[0]['input_ins']
                s_time = time.time()
                # optimize the generator network
                g_loss_gan, g_loss_vgg, g_loss_feat, fake_B_tmp = exe.run(
                    gen_trainer_program,
                    fetch_list=[
                        gen_trainer.gan_loss, gen_trainer.vgg_loss,
                        gen_trainer.gan_feat_loss, gen_trainer.fake_B
                    ],
                    feed={
                        "input_label": data_A,
                        "input_img": data_B,
                        "input_ins": data_C
                    })

                # optimize the discriminator network
                d_loss_real, d_loss_fake = exe.run(
                    dis_trainer_program,
                    fetch_list=[
                        dis_trainer.gan_loss_real, dis_trainer.gan_loss_fake
                    ],
                    feed={
                        "input_label": data_A,
                        "input_img": data_B,
                        "input_ins": data_C,
                        "input_fake": fake_B_tmp
                    })

                batch_time = time.time() - s_time
                t_time += batch_time
                if batch_id % self.cfg.print_freq == 0:
                    print("epoch{}: batch{}: \n\
                         g_loss_gan: {}; g_loss_vgg: {}; g_loss_feat: {} \n\
                         d_loss_real: {}; d_loss_fake: {}; \n\
                         Batch_time_cost: {:.2f}"
                          .format(epoch_id, batch_id, g_loss_gan[0], g_loss_vgg[
                              0], g_loss_feat[0], d_loss_real[0], d_loss_fake[
                                  0], batch_time))

                sys.stdout.flush()
                batch_id += 1
            if self.cfg.run_test:
                test_program = gen_trainer.infer_program
                image_name = fluid.data(
                    name='image_name',
                    shape=[None, self.cfg.batch_size],
                    dtype="int32")
                test_loader = fluid.io.DataLoader.from_generator(
                    feed_list=[input_A, input_B, input_C, image_name],
                    capacity=4,  ## batch_size * 4
                    iterable=True,
                    use_double_buffer=True)
                test_loader.set_batch_generator(
                    self.test_reader,
                    places=fluid.cuda_places()
                    if self.cfg.use_gpu else fluid.cpu_places())
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
            # used for continuous evaluation
            if self.cfg.enable_ce:
                device_num = fluid.core.get_cuda_device_count(
                ) if self.cfg.use_gpu else 1
                print("kpis\tspade_g_loss_gan_card{}\t{}".format(device_num,
                                                                 g_loss_gan[0]))
                print("kpis\tspade_g_loss_vgg_card{}\t{}".format(device_num,
                                                                 g_loss_vgg[0]))
                print("kpis\tspade_g_loss_feat_card{}\t{}".format(
                    device_num, g_loss_feat[0]))
                print("kpis\tspade_d_loss_real_card{}\t{}".format(
                    device_num, d_loss_real[0]))
                print("kpis\tspade_d_loss_fake_card{}\t{}".format(
                    device_num, d_loss_fake[0]))
                print("kpis\tspade_Batch_time_cost_card{}\t{}".format(
                    device_num, batch_time))
