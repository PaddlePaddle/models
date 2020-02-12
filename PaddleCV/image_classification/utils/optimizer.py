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

import math

import paddle.fluid as fluid
import paddle.fluid.layers.ops as ops
from paddle.fluid.layers.learning_rate_scheduler import _decay_step_counter


def cosine_decay(learning_rate, step_each_epoch, epochs=120):
    """Applies cosine decay to the learning rate.
    lr = 0.05 * (math.cos(epoch * (math.pi / 120)) + 1)
    """
    global_step = _decay_step_counter()

    epoch = ops.floor(global_step / step_each_epoch)
    decayed_lr = learning_rate * \
                v(ops.cos(epoch * (math.pi / epochs)) + 1)/2
    return decayed_lr


def cosine_decay_with_warmup(learning_rate,
                             step_each_epoch,
                             epochs=120,
                             warm_up_epoch=5.0):
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
        shape=[1], dtype='float32', value=float(warm_up_epoch), force_cpu=True)

    epoch = ops.floor(global_step / step_each_epoch)
    with fluid.layers.control_flow.Switch() as switch:
        with switch.case(epoch < warmup_epoch):
            decayed_lr = learning_rate * (global_step /
                                          (step_each_epoch * warmup_epoch))
            fluid.layers.tensor.assign(input=decayed_lr, output=lr)
        with switch.default():
            decayed_lr = learning_rate * \
                (ops.cos((global_step - warmup_epoch * step_each_epoch) * (math.pi / ((epochs-warmup_epoch) * step_each_epoch))) + 1)/2
            fluid.layers.tensor.assign(input=decayed_lr, output=lr)
    return lr


def exponential_decay_with_warmup(learning_rate,
                                  step_each_epoch,
                                  decay_epochs,
                                  decay_rate=0.97,
                                  warm_up_epoch=5.0):
    """Applies exponential decay to the learning rate.
    """
    global_step = _decay_step_counter()
    lr = fluid.layers.tensor.create_global_var(
        shape=[1],
        value=0.0,
        dtype='float32',
        persistable=True,
        name="learning_rate")

    warmup_epoch = fluid.layers.fill_constant(
        shape=[1], dtype='float32', value=float(warm_up_epoch), force_cpu=True)

    epoch = ops.floor(global_step / step_each_epoch)
    with fluid.layers.control_flow.Switch() as switch:
        with switch.case(epoch < warmup_epoch):
            decayed_lr = learning_rate * (global_step /
                                          (step_each_epoch * warmup_epoch))
            fluid.layers.assign(input=decayed_lr, output=lr)
        with switch.default():
            div_res = (
                global_step - warmup_epoch * step_each_epoch) / decay_epochs
            div_res = ops.floor(div_res)
            decayed_lr = learning_rate * (decay_rate**div_res)
            fluid.layers.assign(input=decayed_lr, output=lr)

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


class Optimizer(object):
    """A class used to represent several optimizer methods

    Attributes:
        batch_size: batch size on all devices.
        lr: learning rate.
        lr_strategy: learning rate decay strategy.
        l2_decay: l2_decay parameter.
        momentum_rate: momentum rate when using Momentum optimizer.
        step_epochs: piecewise decay steps.
        num_epochs: number of total epochs.

        total_images: total images.
        step: total steps in the an epoch.
        
    """

    def __init__(self, args):
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.lr_strategy = args.lr_strategy
        self.l2_decay = args.l2_decay
        self.momentum_rate = args.momentum_rate
        self.step_epochs = args.step_epochs
        self.num_epochs = args.num_epochs
        self.warm_up_epochs = args.warm_up_epochs
        self.decay_epochs = args.decay_epochs
        self.decay_rate = args.decay_rate
        self.total_images = args.total_images

        self.step = int(math.ceil(float(self.total_images) / self.batch_size))

    def piecewise_decay(self):
        """piecewise decay with Momentum optimizer

            Returns:
            a piecewise_decay optimizer
        """
        bd = [self.step * e for e in self.step_epochs]
        lr = [self.lr * (0.1**i) for i in range(len(bd) + 1)]
        learning_rate = fluid.layers.piecewise_decay(boundaries=bd, values=lr)
        optimizer = fluid.optimizer.Momentum(
            learning_rate=learning_rate,
            momentum=self.momentum_rate,
            regularization=fluid.regularizer.L2Decay(self.l2_decay))
        return optimizer

    def cosine_decay(self):
        """cosine decay with Momentum optimizer

        Returns:
            a cosine_decay optimizer
        """

        learning_rate = fluid.layers.cosine_decay(
            learning_rate=self.lr,
            step_each_epoch=self.step,
            epochs=self.num_epochs)
        optimizer = fluid.optimizer.Momentum(
            learning_rate=learning_rate,
            momentum=self.momentum_rate,
            regularization=fluid.regularizer.L2Decay(self.l2_decay))
        return optimizer

    def cosine_decay_warmup(self):
        """cosine decay with warmup

        Returns:
            a cosine_decay_with_warmup optimizer
        """

        learning_rate = cosine_decay_with_warmup(
            learning_rate=self.lr,
            step_each_epoch=self.step,
            epochs=self.num_epochs,
            warm_up_epoch=self.warm_up_epochs)
        optimizer = fluid.optimizer.Momentum(
            learning_rate=learning_rate,
            momentum=self.momentum_rate,
            regularization=fluid.regularizer.L2Decay(self.l2_decay))
        return optimizer

    def exponential_decay_warmup(self):
        """exponential decay with warmup

        Returns:
            a exponential_decay_with_warmup optimizer
        """

        learning_rate = exponential_decay_with_warmup(
            learning_rate=self.lr,
            step_each_epoch=self.step,
            decay_epochs=self.step * self.decay_epochs,
            decay_rate=self.decay_rate,
            warm_up_epoch=self.warm_up_epochs)
        optimizer = fluid.optimizer.RMSProp(
            learning_rate=learning_rate,
            regularization=fluid.regularizer.L2Decay(self.l2_decay),
            momentum=self.momentum_rate,
            rho=0.9,
            epsilon=0.001)
        return optimizer

    def linear_decay(self):
        """linear decay with Momentum optimizer

        Returns:
            a linear_decay optimizer
        """

        end_lr = 0
        learning_rate = fluid.layers.polynomial_decay(
            self.lr, self.step, end_lr, power=1)
        optimizer = fluid.optimizer.Momentum(
            learning_rate=learning_rate,
            momentum=self.momentum_rate,
            regularization=fluid.regularizer.L2Decay(self.l2_decay))

        return optimizer

    def adam_decay(self):
        """Adam optimizer

        Returns: 
            an adam_decay optimizer
        """

        return fluid.optimizer.Adam(learning_rate=self.lr)

    def cosine_decay_RMSProp(self):
        """cosine decay with RMSProp optimizer

        Returns: 
            an cosine_decay_RMSProp optimizer
        """

        learning_rate = fluid.layers.cosine_decay(
            learning_rate=self.lr,
            step_each_epoch=self.step,
            epochs=self.num_epochs)
        optimizer = fluid.optimizer.RMSProp(
            learning_rate=learning_rate,
            momentum=self.momentum_rate,
            regularization=fluid.regularizer.L2Decay(self.l2_decay),
            # Apply epsilon=1 on ImageNet dataset.
            epsilon=1)
        return optimizer

    def default_decay(self):
        """default decay

        Returns:
            default decay optimizer
        """

        optimizer = fluid.optimizer.Momentum(
            learning_rate=self.lr,
            momentum=self.momentum_rate,
            regularization=fluid.regularizer.L2Decay(self.l2_decay))
        return optimizer


def create_optimizer(args):
    Opt = Optimizer(args)
    optimizer = getattr(Opt, args.lr_strategy)()

    return optimizer
