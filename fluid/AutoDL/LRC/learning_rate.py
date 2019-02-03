from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import paddle
import paddle.fluid as fluid
import paddle.fluid.layers.ops as ops
from paddle.fluid.layers.learning_rate_scheduler import _decay_step_counter
import math
from paddle.fluid.initializer import init_on_cpu


def cosine_decay(learning_rate, num_epoch, steps_one_epoch):
    """Applies cosine decay to the learning rate.
    lr = 0.5 * (math.cos(epoch * (math.pi / 120)) + 1)
    """
    global_step = _decay_step_counter()

    with init_on_cpu():
        decayed_lr = learning_rate * \
                 (ops.cos((global_step / steps_one_epoch) \
                 * math.pi / num_epoch) + 1)/2
    return decayed_lr
