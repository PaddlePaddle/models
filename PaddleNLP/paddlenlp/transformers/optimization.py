# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import sys
import math

from paddle.optimizer.lr import LambdaDecay

__all__ = [
    'LinearSchedulerWithWarmup', 'ConstSchedulerWithWarmup',
    'CosineSchedulerWithWarmup', 'CosineSchedulerWithWarmup',
    'CosineWithHardRestartsScheduleWithWarmup',
    'PolynomialDecaySchedulerWithWarmup'
]


def is_integer(number):
    if sys.version > '3':
        return isinstance(number, int)
    return isinstance(number, (int, long))


class LinearSchedulerWithWarmup(LambdaDecay):
    """
    Create a learning rate scheduler, which increases learning rate linearly
    from 0 to given `learning_rate`, after this warmup period learning rate
    would be decreased linearly from the base learning rate to 0.

    Args:
        learning_rate (float): The base learning rate. It is a python float
            number.
        total_steps (int): The number of training steps.
        warmup (int|float): If int, it means the number of steps for warmup.
            If float, it means the proportion of warmup in total training steps. 
        last_epoch (int, optional): The index of last epoch. It can be set to
            resart training. If None, it means initial learning rate. 
            Default: -1.
        verbose (bool, optional): If True, prints a message to stdout for each
            update. Default: False.

    Examples:
        
        .. code-block:: python

            from paddlenlp.transformers import LinearSchedulerWithWarmup
            lr, warmup_steps, max_steps = 0.1, 100, 1000
            lr_scheduler = LinearSchedulerWithWarmup(lr, max_steps, warmup_steps)

    """

    def __init__(self,
                 learning_rate,
                 total_steps,
                 warmup,
                 last_epoch=-1,
                 verbose=False):
        warmup_steps = warmup if is_integer(warmup) else int(
            math.floor(warmup * total_steps))

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return max(0.0,
                       float(total_steps - current_step) /
                       float(max(1, total_steps - warmup_steps)))

        super(LinearSchedulerWithWarmup, self).__init__(
            learning_rate, lr_lambda, last_epoch, verbose)


class ConstSchedulerWithWarmup(LambdaDecay):
    """
    Create a learning rate scheduler, which increases learning rate linearly
    from 0 to given `learning_rate` during warmup periods and keeps learning
    rate constant.

    Args:
        learning_rate (float): The base learning rate. It is a python float
            number.
        warmup (int): The number of steps for warmup.
        last_epoch (int, optional): The index of last epoch. It can be set to
            resart training. If None, it means initial learning rate. 
            Default: -1.

    Examples:
        
        .. code-block:: python

            from paddlenlp.transformers import ConstSchedulerWithWarmup
            lr, warmup_steps = 0.1, 100
            lr_scheduler = ConstSchedulerWithWarmup(lr, warmup_steps)

    """

    def __init__(self, learning_rate, warmup, last_epoch=-1, verbose=False):
        warmup_steps = warmup

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1.0, warmup_steps))
            return 1.0

        super(ConstSchedulerWithWarmup, self).__init__(learning_rate, lr_lambda,
                                                       last_epoch, verbose)


class CosineSchedulerWithWarmup(LambdaDecay):
    """
    Create a learning rate scheduler, which increases learning rate linearly
    from 0 to given `learning_rate`, after this warmup period learning rate
    would be decreased following the values of the cosine function.

    Args:
        learning_rate (float): The base learning rate. It is a python float
            number.
        total_steps (int): The number of training steps.
        warmup (int|float): If int, it means the number of steps for warmup.
            If float, it means the proportion of warmup in total training steps.
        num_cycles (int, optional): The number of waves in cosine scheduler. If
            None, cosine function returns from the max value to 0. Default: 0.5.
        last_epoch (int, optional): The index of last epoch. It can be set to
            resart training. If None, it means initial learning rate. 
            Default: -1.

    Examples:
        
        .. code-block:: python

            from paddlenlp.transformers import CosineSchedulerWithWarmup
            lr, warmup_steps, max_steps = 0.1, 100, 1000
            lr_scheduler = CosineSchedulerWithWarmup(lr, max_steps, warmup_steps)

    """

    def __init__(self,
                 learning_rate,
                 total_steps,
                 warmup,
                 num_cycles=0.5,
                 last_epoch=-1,
                 verbose=False):
        warmup_steps = warmup if is_integer(warmup) else int(
            math.floor(warmup * total_steps))

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(
                max(1, total_steps - warmup_steps))
            return max(0.0, 0.5 * (
                1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

        super(CosineSchedulerWithWarmup, self).__init__(
            learning_rate, lr_lambda, last_epoch, verbose)


class CosineWithHardRestartsScheduleWithWarmup(LambdaDecay):
    """
    Create a learning rate scheduler, which increases learning rate linearly
    from 0 to given `learning_rate`, after this warmup period learning rate
    would be decreased following the values of the cosine function which has
    serveral hard restarts.

    Args:
        learning_rate (float): The base learning rate. It is a python float
            number.
        total_steps (int): The number of training steps.
        warmup (int|float): If int, it means the number of steps for warmup.
            If float, it means the proportion of warmup in total training steps.
        num_cycles (int, optional): The number of hard restarts to use.
            Default: 1.
        last_epoch (int, optional): The index of last epoch. It can be set to
            resart training. If None, it means initial learning rate. 
            Default: -1.

    Examples:
        
        .. code-block:: python

            from paddlenlp.transformers import CosineWithHardRestartsScheduleWithWarmup
            lr, warmup_steps, max_steps = 0.1, 100, 1000
            lr_scheduler = CosineWithHardRestartsScheduleWithWarmup(
                lr, max_steps, warmup_steps)

    """

    def __init__(self,
                 learning_rate,
                 total_steps,
                 warmup,
                 num_cycles=1,
                 last_epoch=-1,
                 verbose=False):
        warmup_steps = warmup if is_integer(warmup) else int(
            math.floor(warmup * total_steps))

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(
                max(1, total_steps - warmup_steps))
            if progress >= 1.0:
                return 0.0
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * (
                (float(num_cycles) * progress) % 1.0))))

        super(CosineWithHardRestartsScheduleWithWarmup, self).__init__(
            learning_rate, lr_lambda, last_epoch, verbose)


class PolynomialDecaySchedulerWithWarmup(LambdaDecay):
    """
    Create a learning rate scheduler, which increases learning rate linearly
    from 0 to given `lr_init`, after this warmup period learning rate would
    be decreased as a polynomial decay from the base learning rate to the end
    learning rate `lr_end`. 

    Args:
        learning_rate (float): The base learning rate. It is a python float
            number.
        total_steps (int): The number of training steps.
        warmup (int|float): If int, it means the number of steps for warmup.
            If float, it means the proportion of warmup in total training steps.
        lr_end (float, optional): The end learning rate. Default: 1e-7.
        power (float, optional): Power factor. Default: 1.0.
        last_epoch (int, optional): The index of last epoch. It can be set to
            resart training. If None, it means initial learning rate.
            Default: -1.

    Examples:
        
        .. code-block:: python

            from paddlenlp.transformers import PolynomialDecaySchedulerWithWarmup
            lr, lr_end, warmup_steps, max_steps = 0.1, 1e-6, 100, 1000
            lr_scheduler = PolynomialDecaySchedulerWithWarmup(lr, max_steps, warmup_steps, lr_end)

    """

    def __init__(self,
                 learning_rate,
                 total_steps,
                 warmup,
                 lr_end=1e-7,
                 power=1.0,
                 last_epoch=-1,
                 verbose=False):
        lr_init = learning_rate
        assert lr_init > lr_end, f"`lr_end` must be be smaller than `learning_rate`. But `lr_end` is {lr_end} while `learning_rate` is {lr_init}."
        warmup_steps = warmup if is_integer(warmup) else int(
            math.floor(warmup * total_steps))

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            elif current_step > total_steps:
                return lr_end / lr_init  # which multiplies by lr_init equals to lr_end
            else:
                lr_range = lr_init - lr_end
                decay_steps = total_steps - warmup_steps
                pct_remaining = 1 - (current_step - warmup_steps) / decay_steps
                decay = lr_range * pct_remaining**power + lr_end
                return decay / lr_init  # which multiplies by lr_init equals to decay

        super(PolynomialDecaySchedulerWithWarmup, self).__init__(
            lr_init, lr_lambda, last_epoch, verbose)
