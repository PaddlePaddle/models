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
