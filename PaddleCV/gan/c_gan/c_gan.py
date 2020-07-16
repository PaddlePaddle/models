# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
import sys
import os
import six
import argparse
import functools
import matplotlib
import numpy as np
import paddle
import time
import paddle.fluid as fluid
from utility import get_parent_function_name, plot, check, add_arguments, print_arguments
from network import G_cond, D_cond
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

NOISE_SIZE = 100
LEARNING_RATE = 2e-4

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('batch_size',        int,   121,          "Minibatch size.")
add_arg('epoch',             int,   20,        "The number of epoched to be trained.")
add_arg('output',            str,   "./output", "The directory the model and the test result to be saved to.")
add_arg('use_gpu',           bool,  True,       "Whether to use GPU to train.")
add_arg('run_ce',            bool,  False,       "Whether to run for model ce.")
# yapf: enable


def loss(x, label):
    return fluid.layers.mean(
        fluid.layers.sigmoid_cross_entropy_with_logits(
            x=x, label=label))


def train(args):

    if args.run_ce:
        np.random.seed(10)
        fluid.default_startup_program().random_seed = 90

    d_program = fluid.Program()
    dg_program = fluid.Program()

    with fluid.program_guard(d_program):
        conditions = fluid.data(
            name='conditions', shape=[None, 1], dtype='float32')
        img = fluid.data(name='img', shape=[None, 784], dtype='float32')
        label = fluid.data(name='label', shape=[None, 1], dtype='float32')
        d_logit = D_cond(img, conditions)
        d_loss = loss(d_logit, label)

    with fluid.program_guard(dg_program):
        conditions = fluid.data(
            name='conditions', shape=[None, 1], dtype='float32')
        noise = fluid.data(
            name='noise', shape=[None, NOISE_SIZE], dtype='float32')
        g_img = G_cond(z=noise, y=conditions)

        g_program = dg_program.clone()
        g_program_test = dg_program.clone(for_test=True)

        dg_logit = D_cond(g_img, conditions)
        dg_logit_shape = fluid.layers.shape(dg_logit)
        dg_loss = loss(
            dg_logit,
            fluid.layers.fill_constant(
                dtype='float32', shape=[dg_logit_shape[0], 1], value=1.0))

    opt = fluid.optimizer.Adam(learning_rate=LEARNING_RATE)

    opt.minimize(loss=d_loss)
    parameters = [p.name for p in g_program.global_block().all_parameters()]

    opt.minimize(loss=dg_loss, parameter_list=parameters)

    exe = fluid.Executor(fluid.CPUPlace())
    if args.use_gpu:
        exe = fluid.Executor(fluid.CUDAPlace(0))
    exe.run(fluid.default_startup_program())
    if args.run_ce:
        train_reader = fluid.io.batch(
            paddle.dataset.mnist.train(), batch_size=args.batch_size)
    else:
        train_reader = fluid.io.batch(
            fluid.io.shuffle(
                paddle.dataset.mnist.train(), buf_size=60000),
            batch_size=args.batch_size)

    NUM_TRAIN_TIMES_OF_DG = 2
    const_n = np.random.uniform(
        low=-1.0, high=1.0,
        size=[args.batch_size, NOISE_SIZE]).astype('float32')
    t_time = 0
    losses = [[], []]
    for pass_id in range(args.epoch):
        for batch_id, data in enumerate(train_reader()):
            if len(data) != args.batch_size:
                continue
            noise_data = np.random.uniform(
                low=-1.0, high=1.0,
                size=[args.batch_size, NOISE_SIZE]).astype('float32')
            real_image = np.array(list(map(lambda x: x[0], data))).reshape(
                -1, 784).astype('float32')
            conditions_data = np.array([x[1] for x in data]).reshape(
                [-1, 1]).astype("float32")
            real_labels = np.ones(
                shape=[real_image.shape[0], 1], dtype='float32')
            fake_labels = np.zeros(
                shape=[real_image.shape[0], 1], dtype='float32')
            total_label = np.concatenate([real_labels, fake_labels])
            s_time = time.time()
            generated_image = exe.run(
                g_program,
                feed={'noise': noise_data,
                      'conditions': conditions_data},
                fetch_list=[g_img])[0]

            total_images = np.concatenate([real_image, generated_image])

            d_loss_1 = exe.run(d_program,
                               feed={
                                   'img': generated_image,
                                   'label': fake_labels,
                                   'conditions': conditions_data
                               },
                               fetch_list=[d_loss])[0][0]

            d_loss_2 = exe.run(d_program,
                               feed={
                                   'img': real_image,
                                   'label': real_labels,
                                   'conditions': conditions_data
                               },
                               fetch_list=[d_loss])[0][0]

            d_loss_n = d_loss_1 + d_loss_2
            losses[0].append(d_loss_n)
            for _ in six.moves.xrange(NUM_TRAIN_TIMES_OF_DG):
                noise_data = np.random.uniform(
                    low=-1.0, high=1.0,
                    size=[args.batch_size, NOISE_SIZE]).astype('float32')
                dg_loss_n = exe.run(
                    dg_program,
                    feed={'noise': noise_data,
                          'conditions': conditions_data},
                    fetch_list=[dg_loss])[0][0]
                losses[1].append(dg_loss_n)
            batch_time = time.time() - s_time
            t_time += batch_time

            if batch_id % 10 == 0 and not args.run_ce:
                if not os.path.exists(args.output):
                    os.makedirs(args.output)
                # generate image each batch
                generated_images = exe.run(
                    g_program_test,
                    feed={'noise': const_n,
                          'conditions': conditions_data},
                    fetch_list=[g_img])[0]
                total_images = np.concatenate([real_image, generated_images])
                fig = plot(total_images)
                msg = "Epoch ID={0}\n Batch ID={1}\n D-Loss={2}\n DG-Loss={3}\n gen={4}\n " \
                      "Batch_time_cost={5:.2f}".format(
                    pass_id, batch_id, d_loss_n, dg_loss_n, check(generated_images), batch_time)
                print(msg)
                plt.title(msg)
                plt.savefig(
                    '{}/{:04d}_{:04d}.png'.format(args.output, pass_id,
                                                  batch_id),
                    bbox_inches='tight')
                plt.close(fig)

    if args.run_ce:
        print("kpis,cgan_d_train_cost,{}".format(np.mean(losses[0])))
        print("kpis,cgan_g_train_cost,{}".format(np.mean(losses[1])))
        print("kpis,cgan_duration,{}".format(t_time / args.epoch))


if __name__ == "__main__":
    args = parser.parse_args()
    print_arguments(args)
    train(args)
