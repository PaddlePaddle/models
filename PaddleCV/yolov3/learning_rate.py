#  Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
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

import paddle.fluid as fluid
import paddle.fluid.layers.learning_rate_scheduler as lr_scheduler
from paddle.fluid.layers import control_flow


def exponential_with_warmup_decay(learning_rate, boundaries, values,
                                  warmup_iter, warmup_factor):
    global_step = lr_scheduler._decay_step_counter()

    lr = fluid.layers.create_global_var(
        shape=[1],
        value=0.0,
        dtype='float32',
        persistable=True,
        name="learning_rate")

    warmup_iter_var = fluid.layers.fill_constant(
        shape=[1], dtype='float32', value=float(warmup_iter), force_cpu=True)

    with control_flow.Switch() as switch:
        with switch.case(global_step < warmup_iter_var):
            alpha = global_step / warmup_iter_var
            factor = warmup_factor * (1 - alpha) + alpha
            decayed_lr = learning_rate * factor
            fluid.layers.assign(decayed_lr, lr)

        for i in range(len(boundaries)):
            boundary_val = fluid.layers.fill_constant(
                shape=[1],
                dtype='float32',
                value=float(boundaries[i]),
                force_cpu=True)
            value_var = fluid.layers.fill_constant(
                shape=[1], dtype='float32', value=float(values[i]))
            with switch.case(global_step < boundary_val):
                fluid.layers.assign(value_var, lr)

        last_value_var = fluid.layers.fill_constant(
            shape=[1], dtype='float32', value=float(values[len(values) - 1]))
        with switch.default():
            fluid.layers.assign(last_value_var, lr)

    return lr
