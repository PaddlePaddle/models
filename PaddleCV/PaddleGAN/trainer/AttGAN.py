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
            if cfg.epoch <= 100:
                lr = cfg.g_lr
            else:
                lr = fluid.layers.piecewise_decay(
                    boundaries=[99 * step_per_epoch],
                    values=[cfg.g_lr, cfg.g_lr * 0.1], )
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
            clone_image_real = []
            for b in self.program.blocks:
                if b.has_var('image_real'):
                    clone_image_real = b.var('image_real')
                    break
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
                    clone_image_real,
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
                self.d_loss = self.d_loss_real + self.d_loss_fake + self.d_loss_cls

            self.d_loss_real.persistable = True
            self.d_loss_fake.persistable = True
            self.d_loss.persistable = True
            self.d_loss_cls.persistable = True
            self.d_loss_gp.persistable = True
            vars = []
            for var in self.program.list_vars():
                if fluid.io.is_parameter(var) and var.name.startswith(
                        "discriminator"):
                    vars.append(var.name)
            self.param = vars

            if cfg.epoch <= 100:
                lr = cfg.d_lr
            else:
                lr = fluid.layers.piecewise_decay(
                    boundaries=[99 * step_per_epoch],
                    values=[cfg.g_lr, cfg.g_lr * 0.1], )
            optimizer = fluid.optimizer.Adam(
                learning_rate=lr, beta1=0.5, beta2=0.999, name="net_D")

            optimizer.minimize(self.d_loss, parameter_list=vars)

    def gradient_penalty(self, f, real, fake=None, cfg=None, name=None):
        def _interpolate(a, b=None):
            shape = [a.shape[0]]
            alpha = fluid.layers.uniform_random_batch_size_like(
                input=a, shape=shape, min=0.0, max=1.0)
            tmp = fluid.layers.elementwise_mul(
                fluid.layers.elementwise_sub(b, a), alpha, axis=0)
            alpha.stop_gradient = True
            tmp.stop_gradient = True
            inner = fluid.layers.elementwise_add(a, tmp, axis=0)
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

        return parser

    def __init__(self,
                 cfg=None,
                 train_reader=None,
                 test_reader=None,
                 batch_num=1):
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
        gen_trainer = GTrainer(image_real, label_org, label_org_, label_trg,
                               label_trg_, self.cfg, self.batch_num)
        dis_trainer = DTrainer(image_real, label_org, label_org_, label_trg,
                               label_trg_, self.cfg, self.batch_num)

        # prepare environment
        place = fluid.CUDAPlace(0) if self.cfg.use_gpu else fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())

        if self.cfg.init_model:
            utility.init_checkpoints(self.cfg, exe, gen_trainer, "net_G")
            utility.init_checkpoints(self.cfg, exe, dis_trainer, "net_D")

        ### memory optim
        build_strategy = fluid.BuildStrategy()
        build_strategy.enable_inplace = False
        build_strategy.memory_optimize = False

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
            for i in range(self.batch_num):
                image, label_org = next(self.train_reader())
                label_trg = copy.deepcopy(label_org)

                np.random.shuffle(label_trg)
                label_org_ = list(
                    map(lambda x: (x * 2.0 - 1.0) * self.cfg.thres_int,
                        label_org))
                label_trg_ = list(
                    map(lambda x: (x * 2.0 - 1.0) * self.cfg.thres_int,
                        label_trg))

                tensor_img = fluid.LoDTensor()
                tensor_label_org = fluid.LoDTensor()
                tensor_label_trg = fluid.LoDTensor()
                tensor_label_org_ = fluid.LoDTensor()
                tensor_label_trg_ = fluid.LoDTensor()
                tensor_img.set(image, place)
                tensor_label_org.set(label_org, place)
                tensor_label_trg.set(label_trg, place)
                tensor_label_org_.set(label_org_, place)
                tensor_label_trg_.set(label_trg_, place)
                label_shape = tensor_label_trg.shape
                s_time = time.time()
                # optimize the discriminator network
                if (batch_id + 1) % self.cfg.num_discriminator_time != 0:
                    fetches = [
                        dis_trainer.d_loss.name, dis_trainer.d_loss_real.name,
                        dis_trainer.d_loss_fake.name,
                        dis_trainer.d_loss_cls.name, dis_trainer.d_loss_gp.name
                    ]
                    d_loss, d_loss_real, d_loss_fake, d_loss_cls, d_loss_gp = exe.run(
                        dis_trainer_program,
                        fetch_list=fetches,
                        feed={
                            "image_real": tensor_img,
                            "label_org": tensor_label_org,
                            "label_org_": tensor_label_org_,
                            "label_trg": tensor_label_trg,
                            "label_trg_": tensor_label_trg_
                        })

                    batch_time = time.time() - s_time
                    t_time += batch_time
                    print("epoch{}: batch{}:  \n\
                         d_loss: {}; d_loss_real: {}; d_loss_fake: {}; d_loss_cls: {}; d_loss_gp: {} \n\
                         Batch_time_cost: {}".format(epoch_id, batch_id, d_loss[
                        0], d_loss_real[0], d_loss_fake[0], d_loss_cls[0],
                                                     d_loss_gp[0], batch_time))
                # optimize the generator network
                else:
                    d_fetches = [
                        gen_trainer.g_loss_fake.name,
                        gen_trainer.g_loss_rec.name,
                        gen_trainer.g_loss_cls.name, gen_trainer.fake_img.name
                    ]
                    g_loss_fake, g_loss_rec, g_loss_cls, fake_img = exe.run(
                        gen_trainer_program,
                        fetch_list=d_fetches,
                        feed={
                            "image_real": tensor_img,
                            "label_org": tensor_label_org,
                            "label_org_": tensor_label_org_,
                            "label_trg": tensor_label_trg,
                            "label_trg_": tensor_label_trg_
                        })
                    print("epoch{}: batch{}: \n\
                         g_loss_fake: {}; g_loss_rec: {}; g_loss_cls: {}"
                          .format(epoch_id, batch_id, g_loss_fake[0],
                                  g_loss_rec[0], g_loss_cls[0]))
                sys.stdout.flush()
                batch_id += 1

            if self.cfg.run_test:
                test_program = gen_trainer.infer_program
                utility.save_test_image(epoch_id, self.cfg, exe, place,
                                        test_program, gen_trainer,
                                        self.test_reader)

            if self.cfg.save_checkpoints:
                utility.checkpoints(epoch_id, self.cfg, exe, gen_trainer,
                                    "net_G")
                utility.checkpoints(epoch_id, self.cfg, exe, dis_trainer,
                                    "net_D")
