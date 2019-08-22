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
from network.StarGAN_network import StarGAN_model
from util import utility
import paddle.fluid as fluid
import sys
import time
import copy
import numpy as np
import pickle as pkl


class GTrainer():
    def __init__(self, image_real, label_org, label_trg, cfg, step_per_epoch):
        self.program = fluid.default_main_program().clone()
        with fluid.program_guard(self.program):
            model = StarGAN_model()
            self.fake_img = model.network_G(
                image_real, label_trg, cfg, name="g_main")
            self.fake_img.persistable = True
            self.rec_img = model.network_G(
                self.fake_img, label_org, cfg, name="g_main")
            self.rec_img.persistable = True
            self.infer_program = self.program.clone(for_test=False)
            self.g_loss_rec = fluid.layers.reduce_mean(
                fluid.layers.abs(
                    fluid.layers.elementwise_sub(
                        x=image_real, y=self.rec_img)))
            self.pred_fake, self.cls_fake = model.network_D(
                self.fake_img, cfg, name="d_main")
            #wgan
            self.g_loss_fake = -1 * fluid.layers.mean(self.pred_fake)

            cls_shape = self.cls_fake.shape
            self.cls_fake = fluid.layers.reshape(
                self.cls_fake,
                [-1, cls_shape[1] * cls_shape[2] * cls_shape[3]])
            self.g_loss_cls = fluid.layers.reduce_sum(
                fluid.layers.sigmoid_cross_entropy_with_logits(
                    self.cls_fake, label_trg)) / cfg.batch_size
            self.g_loss = self.g_loss_fake + cfg.lambda_rec * self.g_loss_rec + self.g_loss_cls
            self.g_loss_fake.persistable = True
            self.g_loss_rec.persistable = True
            self.g_loss_cls.persistable = True
            lr = cfg.g_lr
            vars = []
            for var in self.program.list_vars():
                if fluid.io.is_parameter(var) and var.name.startswith("g_"):
                    vars.append(var.name)
            self.param = vars
            total_iters = step_per_epoch * cfg.epoch
            boundaries = [cfg.num_iters - cfg.num_iters_decay]
            values = [lr]
            for x in range(cfg.num_iters - cfg.num_iters_decay + 1,
                           total_iters):
                if x % cfg.lr_update_step == 0:
                    boundaries.append(x)
                    lr -= (lr / float(cfg.num_iters_decay))
                    values.append(lr)
            lr = values[-1]
            lr -= (lr / float(cfg.num_iters_decay))
            values.append(lr)
            optimizer = fluid.optimizer.Adam(
                learning_rate=fluid.layers.piecewise_decay(
                    boundaries=boundaries, values=values),
                beta1=0.5,
                beta2=0.999,
                name="net_G")
            optimizer.minimize(self.g_loss, parameter_list=vars)


class DTrainer():
    def __init__(self, image_real, label_org, label_trg, cfg, step_per_epoch):
        self.program = fluid.default_main_program().clone()
        with fluid.program_guard(self.program):
            model = StarGAN_model()

            image_real = fluid.layers.data(
                name='image_real', shape=image_real.shape, dtype='float32')
            self.fake_img = model.network_G(
                image_real, label_trg, cfg, name="g_main")
            self.pred_real, self.cls_real = model.network_D(
                image_real, cfg, name="d_main")
            self.pred_fake, _ = model.network_D(
                self.fake_img, cfg, name="d_main")
            cls_shape = self.cls_real.shape
            self.cls_real = fluid.layers.reshape(
                self.cls_real,
                [-1, cls_shape[1] * cls_shape[2] * cls_shape[3]])
            self.d_loss_cls = fluid.layers.reduce_sum(
                fluid.layers.sigmoid_cross_entropy_with_logits(
                    self.cls_real, label_org)) / cfg.batch_size
            #wgan
            self.d_loss_fake = fluid.layers.mean(self.pred_fake)
            self.d_loss_real = -1 * fluid.layers.mean(self.pred_real)
            self.d_loss_gp = self.gradient_penalty(
                getattr(model, "network_D"),
                image_real,
                self.fake_img,
                cfg=cfg,
                name="d_main")
            self.d_loss = self.d_loss_real + self.d_loss_fake + self.d_loss_cls + cfg.lambda_gp * self.d_loss_gp

            self.d_loss_real.persistable = True
            self.d_loss_fake.persistable = True
            self.d_loss_gp.persistable = True
            self.d_loss_cls.persistable = True
            vars = []
            for var in self.program.list_vars():
                if fluid.io.is_parameter(var) and var.name.startswith("d_"):
                    vars.append(var.name)

            self.param = vars
            total_iters = step_per_epoch * cfg.epoch
            boundaries = [cfg.num_iters - cfg.num_iters_decay]
            values = [cfg.d_lr]
            lr = cfg.d_lr
            for x in range(cfg.num_iters - cfg.num_iters_decay + 1,
                           total_iters):
                if x % cfg.lr_update_step == 0:
                    boundaries.append(x)
                    lr -= (lr / float(cfg.num_iters_decay))
                    values.append(lr)
            lr = values[-1]
            lr -= (lr / float(cfg.num_iters_decay))
            values.append(lr)
            optimizer = fluid.optimizer.Adam(
                learning_rate=fluid.layers.piecewise_decay(
                    boundaries=boundaries, values=values),
                beta1=0.5,
                beta2=0.999,
                name="net_D")

            optimizer.minimize(self.d_loss, parameter_list=vars)

    def gradient_penalty(self, f, real, fake, cfg=None, name=None):
        def _interpolate(a, b):
            shape = [a.shape[0]]
            alpha = fluid.layers.uniform_random_batch_size_like(
                input=a, shape=shape, min=0.0, max=1.0)

            inner = b * (1.0 - alpha) + a * alpha
            return inner

        x = _interpolate(real, fake)
        pred, _ = f(x, cfg, name=name)
        if isinstance(pred, tuple):
            pred = pred[0]
        vars = []
        for var in fluid.default_main_program().list_vars():
            if fluid.io.is_parameter(var) and var.name.startswith('d_'):
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


class StarGAN(object):
    def add_special_args(self, parser):
        parser.add_argument(
            '--g_lr', type=float, default=0.0001, help="learning rate of g")
        parser.add_argument(
            '--d_lr', type=float, default=0.0001, help="learning rate of d")
        parser.add_argument(
            '--c_dim',
            type=int,
            default=5,
            help="the number of attributes we selected")
        parser.add_argument(
            '--g_repeat_num',
            type=int,
            default=6,
            help="number of layers in generator")
        parser.add_argument(
            '--d_repeat_num',
            type=int,
            default=6,
            help="number of layers in discriminator")
        parser.add_argument(
            '--num_iters', type=int, default=200000, help="num iters")
        parser.add_argument(
            '--num_iters_decay',
            type=int,
            default=100000,
            help="num iters decay")
        parser.add_argument(
            '--lr_update_step',
            type=int,
            default=1000,
            help="iters when lr update ")
        parser.add_argument(
            '--lambda_cls',
            type=float,
            default=1.0,
            help="the coefficient of classification")
        parser.add_argument(
            '--lambda_rec',
            type=float,
            default=10.0,
            help="the coefficient of refactor")
        parser.add_argument(
            '--lambda_gp',
            type=float,
            default=10.0,
            help="the coefficient of gradient penalty")
        parser.add_argument(
            '--n_critic',
            type=int,
            default=5,
            help="discriminator training steps when generator update")
        parser.add_argument(
            '--selected_attrs',
            type=str,
            default="Black_Hair,Blond_Hair,Brown_Hair,Male,Young",
            help="the attributes we selected to change")
        parser.add_argument(
            '--n_samples', type=int, default=1, help="batch size when testing")

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

    def build_model(self):
        data_shape = [-1, 3, self.cfg.image_size, self.cfg.image_size]

        image_real = fluid.layers.data(
            name='image_real', shape=data_shape, dtype='float32')
        label_org = fluid.layers.data(
            name='label_org', shape=[self.cfg.c_dim], dtype='float32')
        label_trg = fluid.layers.data(
            name='label_trg', shape=[self.cfg.c_dim], dtype='float32')

        py_reader = fluid.io.PyReader(
            feed_list=[image_real, label_org, label_trg],
            capacity=128,
            iterable=True,
            use_double_buffer=True)

        gen_trainer = GTrainer(image_real, label_org, label_trg, self.cfg,
                               self.batch_num)
        dis_trainer = DTrainer(image_real, label_org, label_trg, self.cfg,
                               self.batch_num)

        # prepare environment
        place = fluid.CUDAPlace(0) if self.cfg.use_gpu else fluid.CPUPlace()
        py_reader.decorate_batch_generator(self.train_reader, places=place)
        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())

        if self.cfg.init_model:
            utility.init_checkpoints(self.cfg, exe, gen_trainer, "net_G")
            utility.init_checkpoints(self.cfg, exe, dis_trainer, "net_D")

        ### memory optim
        build_strategy = fluid.BuildStrategy()
        build_strategy.enable_inplace = False

        gen_trainer_program = fluid.CompiledProgram(
            gen_trainer.program).with_data_parallel(
                loss_name=gen_trainer.g_loss.name,
                build_strategy=build_strategy)
        dis_trainer_program = fluid.CompiledProgram(
            dis_trainer.program).with_data_parallel(
                loss_name=dis_trainer.d_loss.name,
                build_strategy=build_strategy)

        t_time = 0

        for epoch_id in range(self.cfg.epoch):
            batch_id = 0
            for data in py_reader():
                s_time = time.time()
                d_loss_real, d_loss_fake, d_loss, d_loss_cls, d_loss_gp = exe.run(
                    dis_trainer_program,
                    fetch_list=[
                        dis_trainer.d_loss_real, dis_trainer.d_loss_fake,
                        dis_trainer.d_loss, dis_trainer.d_loss_cls,
                        dis_trainer.d_loss_gp
                    ],
                    feed=data)
                # optimize the generator network
                if (batch_id + 1) % self.cfg.n_critic == 0:
                    g_loss_fake, g_loss_rec, g_loss_cls, fake_img, rec_img = exe.run(
                        gen_trainer_program,
                        fetch_list=[
                            gen_trainer.g_loss_fake, gen_trainer.g_loss_rec,
                            gen_trainer.g_loss_cls, gen_trainer.fake_img,
                            gen_trainer.rec_img
                        ],
                        feed=data)
                    print("epoch{}: batch{}: \n\
                         g_loss_fake: {}; g_loss_rec: {}; g_loss_cls: {}"
                          .format(epoch_id, batch_id, g_loss_fake[0],
                                  g_loss_rec[0], g_loss_cls[0]))

                batch_time = time.time() - s_time
                t_time += batch_time
                if (batch_id + 1) % self.cfg.print_freq == 0:
                    print("epoch{}: batch{}: \n\
                         d_loss_real: {}; d_loss_fake: {}; d_loss_cls: {}; d_loss_gp: {} \n\
                         Batch_time_cost: {}".format(
                        epoch_id, batch_id, d_loss_real[0], d_loss_fake[
                            0], d_loss_cls[0], d_loss_gp[0], batch_time))

                sys.stdout.flush()
                batch_id += 1

            if self.cfg.run_test:
                image_name = fluid.layers.data(
                    name='image_name',
                    shape=[self.cfg.n_samples],
                    dtype='int32')
                test_py_reader = fluid.io.PyReader(
                    feed_list=[image_real, label_org, label_trg, image_name],
                    capacity=32,
                    iterable=True,
                    use_double_buffer=True)
                test_py_reader.decorate_batch_generator(
                    self.test_reader, places=place)
                test_program = gen_trainer.infer_program
                utility.save_test_image(epoch_id, self.cfg, exe, place,
                                        test_program, gen_trainer,
                                        test_py_reader)

            if self.cfg.save_checkpoints:
                utility.checkpoints(epoch_id, self.cfg, exe, gen_trainer,
                                    "net_G")
                utility.checkpoints(epoch_id, self.cfg, exe, dis_trainer,
                                    "net_D")
