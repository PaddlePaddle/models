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
"""Optimization and learning rate scheduling."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import paddle.fluid as fluid
import paddle.fluid.layers.learning_rate_scheduler as lr_scheduler
from paddle.fluid.layers import control_flow

import logging
logger = logging.getLogger(__name__)

def cosine_warmup_decay(learning_rate, betas, warmup_factor, decay_factor,
                        total_step, warmup_pct):
    def annealing_cos(start, end, pct):
        "Cosine anneal from `start` to `end` as pct goes from 0.0 to 1.0."
        cos_out = fluid.layers.cos(pct * np.pi) + 1.
        return cos_out * (start - end) / 2. + end

    warmup_start_lr = learning_rate * warmup_factor
    decay_end_lr = learning_rate * decay_factor
    warmup_step = total_step * warmup_pct

    global_step = lr_scheduler._decay_step_counter()

    lr = fluid.layers.create_global_var(
        shape=[1],
        value=float(learning_rate),
        dtype='float32',
        persistable=True,
        name="learning_rate")
    beta1 = fluid.layers.create_global_var(
        shape=[1],
        value=float(betas[0]),
        dtype='float32',
        persistable=True,
        name="beta1")

    warmup_step_var = fluid.layers.fill_constant(
        shape=[1], dtype='float32', value=float(warmup_step), force_cpu=True)

    warmup_pred = global_step < warmup_step_var
    decay_pred = global_step >= warmup_step_var

    # learning rate warmup and decay
    def warmup_lr():
        return annealing_cos(warmup_start_lr, learning_rate,
                             global_step / warmup_step_var)
    def decay_lr():
        return annealing_cos(learning_rate, decay_end_lr,
                (global_step - warmup_step_var) / (total_step - warmup_step))

    lr = fluid.layers.case(pred_fn_pairs=[(warmup_pred, warmup_lr),
                                          (decay_pred, decay_lr)])

    # Adam beta1 warmup and decay
    def warmup_beta1():
        return annealing_cos(betas[0], betas[1],
                             global_step / warmup_step_var)

    def decay_beta1():
        return annealing_cos(betas[0], betas[1],
                             global_step / warmup_step_var)

    beta1 = fluid.layers.case(pred_fn_pairs=[(warmup_pred, warmup_beta1),
                                          (decay_pred, decay_beta1)])

    return lr, beta1


def optimize(loss,
             learning_rate,
             warmup_factor,
             decay_factor,
             total_step,
             warmup_pct,
             train_prog,
             startup_prog,
             weight_decay,
             clip_norm,
             beta1=[0.95, 0.85],
             beta2=0.99,
             scheduler='cosine_warmup_decay'):

    scheduled_lr= None
    if scheduler == 'cosine_warmup_decay':
        scheduled_lr, scheduled_beta1 = cosine_warmup_decay(learning_rate, beta1, warmup_factor,
                                           decay_factor, total_step,
                                           warmup_pct)
    else:
        raise ValueError("Unkown learning rate scheduler, should be "
                         "'cosine_warmup_decay'")

    grad_clip = fluid.clip.GradientClipByGlobalNorm(clip_norm=clip_norm)
    optimizer = fluid.optimizer.Adam(learning_rate=scheduled_lr,
                                     beta1=scheduled_beta1,
                                     beta2=beta2,
                                     grad_clip=grad_clip)

    param_list = dict()

    if weight_decay > 0:
        for param in train_prog.all_parameters():
            param_list[param.name] = param * 1.0
            param_list[param.name].stop_gradient = True

    opt_param_list = []
    for var in train_prog.list_vars():
        if fluid.io.is_parameter(var):
            opt_param_list.append(var.name)
    _, param_grads = optimizer.minimize(loss, parameter_list=opt_param_list)

    if weight_decay > 0:
        for param, grad in param_grads:
            with param.block.program._optimized_guard(
                [param, grad]), fluid.framework.name_scope("weight_decay"):
                updated_param = param - param_list[
                    param.name] * weight_decay * scheduled_lr
                fluid.layers.assign(output=param, input=updated_param)

    return scheduled_lr
