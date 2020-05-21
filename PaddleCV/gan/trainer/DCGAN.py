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
from network.DCGAN_network import DCGAN_model
from util import utility

import sys
import six
import os
import numpy as np
import time
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import paddle.fluid as fluid
import random


class GTrainer():
    def __init__(self, input, label, cfg):
        self.program = fluid.default_main_program().clone()
        with fluid.program_guard(self.program):
            model = DCGAN_model(cfg.batch_size)
            self.fake = model.network_G(input, name='G')
            self.infer_program = self.program.clone(for_test=True)
            d_fake = model.network_D(self.fake, name="D")
            batch = fluid.layers.shape(input)[0]
            fake_labels = fluid.layers.fill_constant(
                dtype='float32', shape=[batch, 1], value=1.0)
            self.g_loss = fluid.layers.reduce_mean(
                fluid.layers.sigmoid_cross_entropy_with_logits(
                    x=d_fake, label=fake_labels))

            vars = []
            for var in self.program.list_vars():
                if fluid.io.is_parameter(var) and (var.name.startswith("G")):
                    vars.append(var.name)
            optimizer = fluid.optimizer.Adam(
                learning_rate=cfg.learning_rate, beta1=0.5, name="net_G")
            optimizer.minimize(self.g_loss, parameter_list=vars)


class DTrainer():
    def __init__(self, input, labels, cfg):
        self.program = fluid.default_main_program().clone()
        with fluid.program_guard(self.program):
            model = DCGAN_model(cfg.batch_size)
            d_logit = model.network_D(input, name="D")
            self.d_loss = fluid.layers.reduce_mean(
                fluid.layers.sigmoid_cross_entropy_with_logits(
                    x=d_logit, label=labels))
            vars = []
            for var in self.program.list_vars():
                if fluid.io.is_parameter(var) and (var.name.startswith("D")):
                    vars.append(var.name)

            optimizer = fluid.optimizer.Adam(
                learning_rate=cfg.learning_rate, beta1=0.5, name="net_D")
            optimizer.minimize(self.d_loss, parameter_list=vars)


class DCGAN(object):
    def add_special_args(self, parser):
        parser.add_argument(
            '--noise_size', type=int, default=100, help="the noise dimension")
        parser.add_argument(
            '--enable_ce',
            action='store_true',
            help="if set, run the tasks with continuous evaluation logs")
        return parser

    def __init__(self, cfg=None, train_reader=None):
        self.cfg = cfg
        self.train_reader = train_reader

    def build_model(self):
        img = fluid.data(name='img', shape=[None, 784], dtype='float32')
        noise = fluid.data(
            name='noise', shape=[None, self.cfg.noise_size], dtype='float32')
        label = fluid.data(name='label', shape=[None, 1], dtype='float32')
        # used for continuous evaluation
        if self.cfg.enable_ce:
            fluid.default_startup_program().random_seed = 90
            random.seed(0)
            np.random.seed(0)

        g_trainer = GTrainer(noise, label, self.cfg)
        d_trainer = DTrainer(img, label, self.cfg)

        # prepare enviorment
        place = fluid.CUDAPlace(0) if self.cfg.use_gpu else fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())

        const_n = np.random.uniform(
            low=-1.0, high=1.0,
            size=[self.cfg.batch_size, self.cfg.noise_size]).astype('float32')

        if self.cfg.init_model:
            utility.init_checkpoints(self.cfg, g_trainer, "net_G")
            utility.init_checkpoints(self.cfg, d_trainer, "net_D")

        ### memory optim
        build_strategy = fluid.BuildStrategy()
        build_strategy.enable_inplace = True

        g_trainer_program = fluid.CompiledProgram(
            g_trainer.program).with_data_parallel(
                loss_name=g_trainer.g_loss.name, build_strategy=build_strategy)
        d_trainer_program = fluid.CompiledProgram(
            d_trainer.program).with_data_parallel(
                loss_name=d_trainer.d_loss.name, build_strategy=build_strategy)

        if self.cfg.run_test:
            image_path = os.path.join(self.cfg.output, 'test')
            if not os.path.exists(image_path):
                os.makedirs(image_path)

        t_time = 0
        for epoch_id in range(self.cfg.epoch):
            for batch_id, data in enumerate(self.train_reader()):
                if len(data) != self.cfg.batch_size:
                    continue

                noise_data = np.random.uniform(
                    low=-1.0,
                    high=1.0,
                    size=[self.cfg.batch_size, self.cfg.noise_size]).astype(
                        'float32')
                real_image = np.array(list(map(lambda x: x[0], data))).reshape(
                    [-1, 784]).astype('float32')
                real_label = np.ones(
                    shape=[real_image.shape[0], 1], dtype='float32')
                fake_label = np.zeros(
                    shape=[real_image.shape[0], 1], dtype='float32')
                s_time = time.time()

                generate_image = exe.run(g_trainer_program,
                                         feed={'noise': noise_data},
                                         fetch_list=[g_trainer.fake])

                d_real_loss = exe.run(
                    d_trainer_program,
                    feed={'img': real_image,
                          'label': real_label},
                    fetch_list=[d_trainer.d_loss])[0]
                d_fake_loss = exe.run(
                    d_trainer_program,
                    feed={'img': generate_image[0],
                          'label': fake_label},
                    fetch_list=[d_trainer.d_loss])[0]
                d_loss = d_real_loss + d_fake_loss

                for _ in six.moves.xrange(self.cfg.num_generator_time):
                    noise_data = np.random.uniform(
                        low=-1.0,
                        high=1.0,
                        size=[self.cfg.batch_size, self.cfg.noise_size]).astype(
                            'float32')
                    g_loss = exe.run(g_trainer_program,
                                     feed={'noise': noise_data},
                                     fetch_list=[g_trainer.g_loss])[0]

                batch_time = time.time() - s_time

                if batch_id % self.cfg.print_freq == 0:
                    print(
                        'Epoch ID: {} Batch ID: {} D_loss: {} G_loss: {} Batch_time_cost: {}'.
                        format(epoch_id, batch_id, d_loss[0], g_loss[0],
                               batch_time))

                t_time += batch_time

                if self.cfg.run_test:
                    generate_const_image = exe.run(
                        g_trainer.infer_program,
                        feed={'noise': const_n},
                        fetch_list=[g_trainer.fake])[0]

                    generate_image_reshape = np.reshape(generate_const_image, (
                        self.cfg.batch_size, -1))
                    total_images = np.concatenate(
                        [real_image, generate_image_reshape])
                    fig = utility.plot(total_images)

                    plt.title('Epoch ID={}, Batch ID={}'.format(epoch_id,
                                                                batch_id))
                    img_name = '{:04d}_{:04d}.png'.format(epoch_id, batch_id)
                    plt.savefig(
                        os.path.join(image_path, img_name), bbox_inches='tight')
                    plt.close(fig)

            if self.cfg.save_checkpoints:
                utility.checkpoints(epoch_id, self.cfg, g_trainer, "net_G")
                utility.checkpoints(epoch_id, self.cfg, d_trainer, "net_D")
        # used for continuous evaluation
        if self.cfg.enable_ce:
            device_num = fluid.core.get_cuda_device_count(
            ) if self.cfg.use_gpu else 1
            print("kpis\tdcgan_d_loss_card{}\t{}".format(device_num, d_loss[0]))
            print("kpis\tdcgan_g_loss_card{}\t{}".format(device_num, g_loss[0]))
            print("kpis\tdcgan_Batch_time_cost_card{}\t{}".format(device_num,
                                                                  batch_time))
