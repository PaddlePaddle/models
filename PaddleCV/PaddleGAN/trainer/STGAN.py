from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from network.STGAN_network import STGAN_model
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
            model = STGAN_model()
            self.fake_img, self.rec_img = model.network_G(
                image_real, label_org_, label_trg_, cfg, name="generator")
            self.fake_img.persistable = True
            self.rec_img.persistable = True
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
                ones = fluid.layers.fill_constant_batch_size_like(
                    input=self.pred_fake,
                    shape=self.pred_fake.shape,
                    value=1.0,
                    dtype='float32')
                self.g_loss_fake = fluid.layers.mean(
                    fluid.layers.square(
                        fluid.layers.elementwise_sub(
                            x=self.pred_fake, y=ones)))

            self.g_loss_cls = fluid.layers.mean(
                fluid.layers.sigmoid_cross_entropy_with_logits(self.cls_fake,
                                                               label_trg))
            self.g_loss = self.g_loss_fake + cfg.lambda_rec * self.g_loss_rec + cfg.lambda_cls * self.g_loss_cls
            self.g_loss_fake.persistable = True
            self.g_loss_rec.persistable = True
            self.g_loss_cls.persistable = True
            lr = cfg.g_lr
            vars = []
            for var in self.program.list_vars():
                if fluid.io.is_parameter(var) and var.name.startswith(
                        "generator"):
                    vars.append(var.name)
            self.param = vars
            optimizer = fluid.optimizer.Adam(
                learning_rate=fluid.layers.piecewise_decay(
                    boundaries=[99 * step_per_epoch], values=[lr, lr * 0.1]),
                beta1=0.5,
                beta2=0.999,
                name="net_G")

            optimizer.minimize(self.g_loss, parameter_list=vars)


class DTrainer():
    def __init__(self, image_real, label_org, label_org_, label_trg, label_trg_,
                 cfg, step_per_epoch):
        self.program = fluid.default_main_program().clone()
        lr = cfg.d_lr
        with fluid.program_guard(self.program):
            model = STGAN_model()
            self.fake_img, _ = model.network_G(
                image_real, label_org_, label_trg_, cfg, name="generator")
            self.pred_real, self.cls_real = model.network_D(
                image_real, cfg, name="discriminator")
            self.pred_real.persistable = True
            self.cls_real.persistable = True
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
                ones = fluid.layers.fill_constant_batch_size_like(
                    input=self.pred_real,
                    shape=self.pred_real.shape,
                    value=1.0,
                    dtype='float32')
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

            self.d_loss_real.persistable = True
            self.d_loss_fake.persistable = True
            self.d_loss.persistable = True
            self.d_loss_cls.persistable = True
            self.d_loss_gp.persistable = True
            vars = []
            for var in self.program.list_vars():
                if fluid.io.is_parameter(var) and (
                        var.name.startswith("discriminator")):
                    vars.append(var.name)
            self.param = vars

            optimizer = fluid.optimizer.Adam(
                learning_rate=fluid.layers.piecewise_decay(
                    boundaries=[99 * step_per_epoch],
                    values=[lr, lr * 0.1], ),
                beta1=0.5,
                beta2=0.999,
                name="net_D")

            optimizer.minimize(self.d_loss, parameter_list=vars)
            f = open('G_program.txt', 'w')
            print(self.program, file=f)

    def gradient_penalty(self, f, real, fake=None, cfg=None, name=None):
        def _interpolate(a, b=None):
            if b is None:
                beta = fluid.layers.uniform_random_batch_size_like(
                    input=a, shape=a.shape, min=0.0, max=1.0)
                mean = fluid.layers.reduce_mean(
                    a, range(len(a.shape)), keep_dim=True)
                input_sub_mean = fluid.layers.elementwise_sub(a, mean, axis=0)
                var = fluid.layers.reduce_mean(
                    fluid.layers.square(input_sub_mean),
                    range(len(a.shape)),
                    keep_dim=True)
                b = beta * fluid.layers.sqrt(var) * 0.5 + a
            shape = [a.shape[0]]
            alpha = fluid.layers.uniform_random_batch_size_like(
                input=a, shape=shape, min=0.0, max=1.0)
            inner = (b - a) * alpha + a
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


class STGAN(object):
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
            '--use_gru', type=bool, default=True, help="whether to use GRU")
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
            help="default layers in generotor")
        parser.add_argument(
            '--gru_n_layers',
            type=int,
            default=4,
            help="default layers of GRU in generotor")
        parser.add_argument(
            '--dis_norm',
            type=str,
            default=None,
            help="the normalization in discriminator, choose in [None, instance_norm]"
        )

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
        label_org_ = fluid.layers.data(
            name='label_org_', shape=[self.cfg.c_dim], dtype='float32')
        label_trg_ = fluid.layers.data(
            name='label_trg_', shape=[self.cfg.c_dim], dtype='float32')

        test_gen_trainer = GTrainer(image_real, label_org, label_org_,
                                    label_trg, label_trg_, self.cfg,
                                    self.batch_num)

        py_reader = fluid.io.PyReader(
            feed_list=[image_real, label_org, label_trg],
            capacity=64,
            iterable=True,
            use_double_buffer=True)
        label_org_ = (label_org * 2.0 - 1.0) * self.cfg.thres_int
        label_trg_ = (label_trg * 2.0 - 1.0) * self.cfg.thres_int

        gen_trainer = GTrainer(image_real, label_org, label_org_, label_trg,
                               label_trg_, self.cfg, self.batch_num)
        dis_trainer = DTrainer(image_real, label_org, label_org_, label_trg,
                               label_trg_, self.cfg, self.batch_num)

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
                # optimize the discriminator network
                fetches = [
                    dis_trainer.d_loss.name,
                    dis_trainer.d_loss_real.name,
                    dis_trainer.d_loss_fake.name,
                    dis_trainer.d_loss_cls.name,
                    dis_trainer.d_loss_gp.name,
                ]
                d_loss, d_loss_real, d_loss_fake, d_loss_cls, d_loss_gp, = exe.run(
                    dis_trainer_program, fetch_list=fetches, feed=data)
                if (batch_id + 1) % self.cfg.num_discriminator_time == 0:
                    # optimize the generator network
                    d_fetches = [
                        gen_trainer.g_loss_fake.name,
                        gen_trainer.g_loss_rec.name, gen_trainer.g_loss_cls.name
                    ]
                    g_loss_fake, g_loss_rec, g_loss_cls = exe.run(
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
                test_program = test_gen_trainer.infer_program
                utility.save_test_image(epoch_id, self.cfg, exe, place,
                                        test_program, test_gen_trainer,
                                        test_py_reader)

            if self.cfg.save_checkpoints:
                utility.checkpoints(epoch_id, self.cfg, exe, gen_trainer,
                                    "net_G")
                utility.checkpoints(epoch_id, self.cfg, exe, dis_trainer,
                                    "net_D")
