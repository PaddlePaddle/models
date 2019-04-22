from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import paddle
import paddle.fluid as fluid
import paddle.fluid.layers.ops as ops
from paddle.fluid.initializer import init_on_cpu
from paddle.fluid.layers.learning_rate_scheduler import _decay_step_counter
import math


def cosine_decay(learning_rate, step_each_epoch, epochs=120):
    """Applies cosine decay to the learning rate.
    lr = 0.05 * (math.cos(epoch * (math.pi / 120)) + 1)
    """
    global_step = _decay_step_counter()

    with init_on_cpu():
        epoch = ops.floor(global_step / step_each_epoch)
        decayed_lr = learning_rate * \
                     (ops.cos(epoch * (math.pi / epochs)) + 1)/2
    return decayed_lr

def cosine_decay_with_warmup(learning_rate, step_each_epoch, epochs=120):
    """Applies cosine decay to the learning rate.
    lr = 0.05 * (math.cos(epoch * (math.pi / 120)) + 1)
    decrease lr for every mini-batch and start with warmup.
    """
    global_step = _decay_step_counter()
    lr = fluid.layers.tensor.create_global_var(
    shape=[1],
    value=0.0,
    dtype='float32',
    persistable=True,
    name="learning_rate")

    warmup_epoch = fluid.layers.fill_constant(
        shape=[1], dtype='float32', value=float(5), force_cpu=True)

    with init_on_cpu():
        epoch = ops.floor(global_step / step_each_epoch)
        with control_flow.Switch() as switch:
            with switch.case(epoch < warmup_epoch):
                decayed_lr = learning_rate * (global_step / (step_each_epoch * warmup_epoch))
                fluid.layers.tensor.assign(input=decayed_lr, output=lr)
            with switch.default():
                decayed_lr = learning_rate * \
                    (ops.cos((global_step - warmup_epoch * step_each_epoch) * (math.pi / (epochs * step_each_epoch))) + 1)/2
                fluid.layers.tensor.assign(input=decayed_lr, output=lr)
    return lr

def lr_warmup(learning_rate, warmup_steps, start_lr, end_lr):
    """ Applies linear learning rate warmup for distributed training
        Argument learning_rate can be float or a Variable
        lr = lr + (warmup_rate * step / warmup_steps)
    """
    assert (isinstance(end_lr, float))
    assert (isinstance(start_lr, float))
    linear_step = end_lr - start_lr
    with fluid.default_main_program()._lr_schedule_guard():
        lr = fluid.layers.tensor.create_global_var(
            shape=[1],
            value=0.0,
            dtype='float32',
            persistable=True,
            name="learning_rate_warmup")

        global_step = fluid.layers.learning_rate_scheduler._decay_step_counter()

        with fluid.layers.control_flow.Switch() as switch:
            with switch.case(global_step < warmup_steps):
                decayed_lr = start_lr + linear_step * (global_step /
                                                       warmup_steps)
                fluid.layers.tensor.assign(decayed_lr, lr)
            with switch.default():
                fluid.layers.tensor.assign(learning_rate, lr)

        return lr
