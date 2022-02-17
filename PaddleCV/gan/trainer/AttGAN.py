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
from network.AttGAN_network import AttGAN_model
from util import utility
import paddle.fluid as fluid
import sys
import time
import copy
import numpy as np


class GTrainer():
    def __init__(self, image_real, label_org, label_org_, label_trg, label_trg_,
                 cfg, step_per_epoch):
        self.program = fluid.default_main_program().clone()
        with fluid.program_guard(self.program):
            model = AttGAN_model()
            self.fake_img, self.rec_img = model.network_G(
                image_real, label_org_, label_trg_, cfg, name="generator")
            self.infer_program = self.program.clone(for_test=True)

            self.g_loss_rec = fluid.layers.mean(
                fluid.layers.abs(
                    fluid.layers.elementwise_sub(
                        x=image_real, y=self.rec_img)))
            self.pred_fake, self.cls_fake = model.network_D(
                self.fake_img, cfg, name="discriminator")
            #wgan
            if cfg.gan_mode == "wgan":
                self.g_loss_fake = -1 * fluid.layers.mean(self.pred_fake)
            #lsgan
            elif cfg.gan_mode == "lsgan":
                fake_shape = fluid.layers.shape(self.pred_fake)
                ones = fluid.layers.fill_constant(
                    shape=fake_shape, value=1.0, dtype='float32')
                self.g_loss_fake = fluid.layers.mean(
                    fluid.layers.square(
                        fluid.layers.elementwise_sub(
                            x=self.pred_fake, y=ones)))
            else:
                raise NotImplementedError("gan_mode {} is not support!".format(
                    cfg.gan_mode))

            self.g_loss_cls = fluid.layers.mean(
                fluid.layers.sigmoid_cross_entropy_with_logits(self.cls_fake,
                                                               label_trg))
            self.g_loss = self.g_loss_fake + cfg.lambda_rec * self.g_loss_rec + cfg.lambda_cls * self.g_loss_cls

            lr = fluid.layers.piecewise_decay(
                boundaries=[99 * step_per_epoch],
                values=[cfg.g_lr, cfg.g_lr * 0.1])
            vars = []
            for var in self.program.list_vars():
                if fluid.io.is_parameter(var) and var.name.startswith(
                        "generator"):
                    vars.append(var.name)
            self.param = vars
            optimizer = fluid.optimizer.Adam(
                learning_rate=lr, beta1=0.5, beta2=0.999, name="net_G")

            optimizer.minimize(self.g_loss, parameter_list=vars)


class DTrainer():
    def __init__(self, image_real, label_org, label_org_, label_trg, label_trg_,
                 cfg, step_per_epoch):
        self.program = fluid.default_main_program().clone()
        lr = cfg.d_lr
        with fluid.program_guard(self.program):
            model = AttGAN_model()
            self.fake_img, _ = model.network_G(
                image_real, label_org, label_trg_, cfg, name="generator")
            self.pred_real, self.cls_real = model.network_D(
                image_real, cfg, name="discriminator")
            self.pred_fake, _ = model.network_D(
                self.fake_img, cfg, name="discriminator")
            self.d_loss_cls = fluid.layers.mean(
                fluid.layers.sigmoid_cross_entropy_with_logits(self.cls_real,
                                                               label_org))
            #wgan
            if cfg.gan_mode == "wgan":
                self.d_loss_fake = fluid.layers.reduce_mean(self.pred_fake)
                self.d_loss_real = -1 * fluid.layers.reduce_mean(self.pred_real)
                self.d_loss_gp = self.gradient_penalty(
                    model.network_D,
                    image_real,
                    self.fake_img,
                    cfg=cfg,
                    name="discriminator")
                self.d_loss = self.d_loss_real + self.d_loss_fake + 1.0 * self.d_loss_cls + cfg.lambda_gp * self.d_loss_gp
            #lsgan
            elif cfg.gan_mode == "lsgan":
                real_shape = fluid.layers.shape(self.pred_real)
                ones = fluid.layers.fill_constant(
                    shape=real_shape, value=1.0, dtype='float32')
                self.d_loss_real = fluid.layers.mean(
                    fluid.layers.square(
                        fluid.layers.elementwise_sub(
                            x=self.pred_real, y=ones)))
                self.d_loss_fake = fluid.layers.mean(
                    fluid.layers.square(x=self.pred_fake))
                self.d_loss_gp = self.gradient_penalty(
                    model.network_D,
                    image_real,
                    None,
                    cfg=cfg,
                    name="discriminator")
                self.d_loss = self.d_loss_real + self.d_loss_fake + 1.0 * self.d_loss_cls + cfg.lambda_gp * self.d_loss_gp
            else:
                raise NotImplementedError("gan_mode {} is not support!".format(
                    cfg.gan_mode))

            vars = []
            for var in self.program.list_vars():
                if fluid.io.is_parameter(var) and var.name.startswith(
                        "discriminator"):
                    vars.append(var.name)
            self.param = vars

            lr = fluid.layers.piecewise_decay(
                boundaries=[99 * step_per_epoch],
                values=[cfg.g_lr, cfg.g_lr * 0.1])
            optimizer = fluid.optimizer.Adam(
                learning_rate=lr, beta1=0.5, beta2=0.999, name="net_D")

            optimizer.minimize(self.d_loss, parameter_list=vars)

    def gradient_penalty(self, f, real, fake=None, cfg=None, name=None):
        def _interpolate(a, b=None):
            a_shape = fluid.layers.shape(a)
            if b is None:
                if cfg.enable_ce:
                    beta = fluid.layers.uniform_random(
                        shape=a_shape, min=0.0, max=1.0, seed=1)
                else:
                    beta = fluid.layers.uniform_random(
                        shape=a_shape, min=0.0, max=1.0)

                mean = fluid.layers.reduce_mean(
                    a, dim=list(range(len(a.shape))))
                input_sub_mean = fluid.layers.elementwise_sub(a, mean, axis=0)
                var = fluid.layers.reduce_mean(
                    fluid.layers.square(input_sub_mean),
                    dim=list(range(len(a.shape))))
                b = beta * fluid.layers.sqrt(var) * 0.5 + a
            shape = [a.shape[0]]
            if cfg.enable_ce:
                alpha = fluid.layers.uniform_random(
                    shape=a_shape[0], min=0.0, max=1.0, seed=1)
            else:
                alpha = fluid.layers.uniform_random(
                    shape=a_shape[0], min=0.0, max=1.0)

            inner = fluid.layers.elementwise_mul((b - a), alpha, axis=0) + a
            return inner

        x = _interpolate(real, fake)

        pred, _ = f(x, cfg=cfg, name=name)
        if isinstance(pred, tuple):
            pred = pred[0]
        vars = []
        for var in fluid.default_main_program().list_vars():
            if fluid.io.is_parameter(var) and var.name.startswith(
                    "discriminator"):
                vars.append(var.name)
        grad = fluid.gradients(pred, x, no_grad_set=vars)[0]
        grad_shape = grad.shape
        grad = fluid.layers.reshape(
            grad, [-1, grad_shape[1] * grad_shape[2] * grad_shape[3]])
        epsilon = 1e-16
        norm = fluid.layers.sqrt(
            fluid.layers.reduce_sum(
                fluid.layers.square(grad), dim=1) + epsilon)
        gp = fluid.layers.reduce_mean(fluid.layers.square(norm - 1.0))
        return gp


class AttGAN(object):
    def add_special_args(self, parser):
        parser.add_argument(
            '--g_lr',
            type=float,
            default=0.0002,
            help="the base learning rate of generator")
        parser.add_argument(
            '--d_lr',
            type=float,
            default=0.0002,
            help="the base learning rate of discriminator")
        parser.add_argument(
            '--c_dim',
            type=int,
            default=13,
            help="the number of attributes we selected")
        parser.add_argument(
            '--d_fc_dim',
            type=int,
            default=1024,
            help="the base fc dim in discriminator")
        parser.add_argument(
            '--lambda_cls',
            type=float,
            default=10.0,
            help="the coefficient of classification")
        parser.add_argument(
            '--lambda_rec',
            type=float,
            default=100.0,
            help="the coefficient of refactor")
        parser.add_argument(
            '--thres_int',
            type=float,
            default=0.5,
            help="thresh change of attributes")
        parser.add_argument(
            '--lambda_gp',
            type=float,
            default=10.0,
            help="the coefficient of gradient penalty")
        parser.add_argument(
            '--n_samples', type=int, default=16, help="batch size when testing")
        parser.add_argument(
            '--selected_attrs',
            type=str,
            default="Bald,Bangs,Black_Hair,Blond_Hair,Brown_Hair,Bushy_Eyebrows,Eyeglasses,Male,Mouth_Slightly_Open,Mustache,No_Beard,Pale_Skin,Young",
            help="the attributes we selected to change")
        parser.add_argument(
            '--n_layers',
            type=int,
            default=5,
            help="default layers in the network")
        parser.add_argument(
            '--dis_norm',
            type=str,
            default=None,
            help="the normalization in discriminator, choose in [None, instance_norm]"
        )
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
        data_shape = [None, 3, self.cfg.image_size, self.cfg.image_size]

        image_real = fluid.data(
            name='image_real', shape=data_shape, dtype='float32')
        label_org = fluid.data(
            name='label_org', shape=[None, self.cfg.c_dim], dtype='float32')
        label_trg = fluid.data(
            name='label_trg', shape=[None, self.cfg.c_dim], dtype='float32')
        label_org_ = fluid.data(
            name='label_org_', shape=[None, self.cfg.c_dim], dtype='float32')
        label_trg_ = fluid.data(
            name='label_trg_', shape=[None, self.cfg.c_dim], dtype='float32')
        # used for continuous evaluation        
        if self.cfg.enable_ce:
            fluid.default_startup_program().random_seed = 90

        loader = fluid.io.DataLoader.from_generator(
            feed_list=[image_real, label_org, label_trg],
            capacity=64,
            iterable=True,
            use_double_buffer=True)

        test_gen_trainer = GTrainer(image_real, label_org, label_org_,
                                    label_trg, label_trg_, self.cfg,
                                    self.batch_num)

        label_org_ = (label_org * 2.0 - 1.0) * self.cfg.thres_int
        label_trg_ = (label_trg * 2.0 - 1.0) * self.cfg.thres_int

        gen_trainer = GTrainer(image_real, label_org, label_org_, label_trg,
                               label_trg_, self.cfg, self.batch_num)
        dis_trainer = DTrainer(image_real, label_org, label_org_, label_trg,
                               label_trg_, self.cfg, self.batch_num)

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
        # used for continuous evaluation        
        if self.cfg.enable_ce:
            gen_trainer_program.random_seed = 90
            dis_trainer_program.random_seed = 90

        t_time = 0

        for epoch_id in range(self.cfg.epoch):
            batch_id = 0
            for data in loader():
                s_time = time.time()
                # optimize the discriminator network
                fetches = [
                    dis_trainer.d_loss.name,
                    dis_trainer.d_loss_real.name,
                    dis_trainer.d_loss_fake.name,
                    dis_trainer.d_loss_cls.name,
                    dis_trainer.d_loss_gp.name,
                ]
                d_loss, d_loss_real, d_loss_fake, d_loss_cls, d_loss_gp = exe.run(
                    dis_trainer_program, fetch_list=fetches, feed=data)

                if (batch_id + 1) % self.cfg.num_discriminator_time == 0:
                    # optimize the generator network
                    d_fetches = [
                        gen_trainer.g_loss_fake.name,
                        gen_trainer.g_loss_rec.name,
                        gen_trainer.g_loss_cls.name, gen_trainer.fake_img.name
                    ]
                    g_loss_fake, g_loss_rec, g_loss_cls, fake_img = exe.run(
                        gen_trainer_program, fetch_list=d_fetches, feed=data)
                    print("epoch{}: batch{}: \n\
                         g_loss_fake: {}; g_loss_rec: {}; g_loss_cls: {}"
                          .format(epoch_id, batch_id, g_loss_fake[0],
                                  g_loss_rec[0], g_loss_cls[0]))

                batch_time = time.time() - s_time
                t_time += batch_time
                if (batch_id + 1) % self.cfg.print_freq == 0:
                    print("epoch{}: batch{}:  \n\
                         d_loss: {}; d_loss_real: {}; d_loss_fake: {}; d_loss_cls: {}; d_loss_gp: {} \n\
                         Batch_time_cost: {}".format(epoch_id, batch_id, d_loss[
                        0], d_loss_real[0], d_loss_fake[0], d_loss_cls[0],
                                                     d_loss_gp[0], batch_time))
                sys.stdout.flush()
                batch_id += 1
                if self.cfg.enable_ce and batch_id == 100:
                    break

            if self.cfg.run_test:
                image_name = fluid.data(
                    name='image_name',
                    shape=[None, self.cfg.n_samples],
                    dtype='int32')
                test_loader = fluid.io.DataLoader.from_generator(
                    feed_list=[image_real, label_org, label_trg, image_name],
                    capacity=32,
                    iterable=True,
                    use_double_buffer=True)
                test_loader.set_batch_generator(
                    self.test_reader,
                    places=fluid.cuda_places()
                    if self.cfg.use_gpu else fluid.cpu_places())

                test_program = test_gen_trainer.infer_program
                utility.save_test_image(epoch_id, self.cfg, exe, place,
                                        test_program, test_gen_trainer,
                                        test_loader)

            if self.cfg.save_checkpoints:
                utility.checkpoints(epoch_id, self.cfg, gen_trainer, "net_G")
                utility.checkpoints(epoch_id, self.cfg, dis_trainer, "net_D")
            # used for continuous evaluation
            if self.cfg.enable_ce:
                device_num = fluid.core.get_cuda_device_count(
                ) if self.cfg.use_gpu else 1
                print("kpis\tattgan_g_loss_fake_card{}\t{}".format(
                    device_num, g_loss_fake[0]))
                print("kpis\tattgan_g_loss_rec_card{}\t{}".format(
                    device_num, g_loss_rec[0]))
                print("kpis\tattgan_g_loss_cls_card{}\t{}".format(
                    device_num, g_loss_cls[0]))
                print("kpis\tattgan_d_loss_real_card{}\t{}".format(
                    device_num, d_loss_real[0]))
                print("kpis\tattgan_d_loss_fake_card{}\t{}".format(
                    device_num, d_loss_fake[0]))
                print("kpis\tattgan_d_loss_gp_card{}\t{}".format(device_num,
                                                                 d_loss_gp[0]))
                print("kpis\tattgan_Batch_time_cost_card{}\t{}".format(
                    device_num, batch_time))
