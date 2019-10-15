# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved
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

import paddle.fluid as fluid
import utility
import time
import numpy as np
from model_search import model


def compute_unrolled_step(image_train, label_train, image_val, label_val,
                          data_prog, startup_prog, lr, args):

    fetch = []
    unrolled_model_prog = data_prog.clone()
    with fluid.program_guard(unrolled_model_prog, startup_prog):
        # construct model graph
        train_logits, train_loss = model(
            image_train,
            label_train,
            args.init_channels,
            args.class_num,
            args.layers,
            name="model")
        # construct unrolled model graph
        logits, unrolled_train_loss = model(
            image_train,
            label_train,
            args.init_channels,
            args.class_num,
            args.layers,
            name="unrolled_model")

        all_params = unrolled_model_prog.global_block().all_parameters()
        model_var = utility.get_parameters(all_params, 'model')[1]
        unrolled_model_var = utility.get_parameters(all_params,
                                                    'unrolled_model')[1]

        # copy model_var to unrolled_model_var
        for m_var, um_var in zip(model_var, unrolled_model_var):
            fluid.layers.assign(m_var, um_var)

        unrolled_optimizer = fluid.optimizer.MomentumOptimizer(
            lr,
            args.momentum,
            regularization=fluid.regularizer.L2DecayRegularizer(
                args.weight_decay))
        unrolled_optimizer.minimize(
            unrolled_train_loss,
            parameter_list=[v.name for v in unrolled_model_var])
        fetch.append(unrolled_train_loss)
    print(
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "get unrolled_model")

    arch_optim_prog = data_prog.clone()
    with fluid.program_guard(arch_optim_prog, startup_prog):
        train_logits, train_loss = model(
            image_train,
            label_train,
            args.init_channels,
            args.class_num,
            args.layers,
            name="model")
        logits, unrolled_valid_loss = model(
            image_val,
            label_val,
            args.init_channels,
            args.class_num,
            args.layers,
            name="unrolled_model")

        all_params = arch_optim_prog.global_block().all_parameters()
        model_var = utility.get_parameters(all_params, 'model')[1]
        unrolled_model_var = utility.get_parameters(all_params,
                                                    'unrolled_model')[1]
        arch_var = utility.get_parameters(all_params, 'arch')[1]
        # get grad of unrolled_valid_loss
        valid_grads = fluid.gradients(unrolled_valid_loss, unrolled_model_var)
        eps = 1e-2 * fluid.layers.rsqrt(
            fluid.layers.sums([
                fluid.layers.reduce_sum(fluid.layers.square(valid_grad))
                for valid_grad in valid_grads
            ]))
        model_params_grads = list(zip(model_var, valid_grads))

        for param, grad in model_params_grads:
            param = fluid.layers.elementwise_add(
                param, fluid.layers.elementwise_mul(grad, eps))
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "get w+")

        logits, train_loss = model(
            image_train,
            label_train,
            args.init_channels,
            args.class_num,
            args.layers,
            name="model")
        train_grads_pos = fluid.gradients(train_loss, arch_var)
        grads_names = [v.name for v in train_grads_pos]
        for name in grads_names:
            arch_optim_prog.global_block()._rename_var(name, name + '_pos')
        print(
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "get train_gards_pos")

        # w- = w - eps*dw`"""
        for param, grad in model_params_grads:
            param = fluid.layers.elementwise_add(
                param, fluid.layers.elementwise_mul(grad, eps * -2))
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "get w-")

        logits, train_loss = model(
            image_train,
            label_train,
            args.init_channels,
            args.class_num,
            args.layers,
            name="model")
        train_grads_neg = fluid.gradients(train_loss, arch_var)
        for name in grads_names:
            arch_optim_prog.global_block()._rename_var(name, name + '_neg')
        print(
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "get train_gards_neg")

        # recover w
        for param, grad in model_params_grads:
            param = fluid.layers.elementwise_add(
                param, fluid.layers.elementwise_mul(grad, eps))
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "get w")

        leader_opt = fluid.optimizer.Adam(
            args.arch_learning_rate,
            0.5,
            0.999,
            regularization=fluid.regularizer.L2DecayRegularizer(
                args.arch_weight_decay))
        arch_params_grads = leader_opt.backward(
            unrolled_valid_loss, parameter_list=[v.name for v in arch_var])

        grads_p = [
            arch_optim_prog.global_block().var(name + '_pos')
            for name in grads_names
        ]
        grads_n = [
            arch_optim_prog.global_block().var(name + '_neg')
            for name in grads_names
        ]

        for i, (var, grad) in enumerate(arch_params_grads):
            arch_params_grads[i] = (var, grad - (
                (grads_p[i] - grads_n[i]) / (eps * 2)) * lr)
        leader_opt.apply_gradients(arch_params_grads)
        print(
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "update alpha")
        fetch.append(unrolled_valid_loss)
        arch_progs_list = [unrolled_model_prog, arch_optim_prog]
    return arch_progs_list, fetch
