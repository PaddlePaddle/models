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
    'LinearDecayWithWarmUp', 'ConstSchedulerWithWarmup',
    'CosineDecayWithWarmUp', 'PolyDecayWithWarmUp'
]


def is_integer(number):
    if sys.version > '3':
        return isinstance(number, int)
    return isinstance(number, (int, long))


class LinearDecayWithWarmUp(LambdaDecay):
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
            restart training. If None, it means initial learning rate. 
            Default: -1.
        verbose (bool, optional): If True, prints a message to stdout for each
            update. Default: False.

    Examples:
        
        .. code-block:: python

            from paddlenlp.transformers import LinearDecayWithWarmUp
            lr, warmup_steps, max_steps = 0.1, 100, 1000
            lr_scheduler = LinearDecayWithWarmUp(lr, max_steps, warmup_steps)

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

        super(LinearDecayWithWarmUp, self).__init__(learning_rate, lr_lambda,
                                                    last_epoch, verbose)


class ConstSchedulerWithWarmup(LambdaDecay):
    """
    Create a learning rate scheduler, which increases learning rate linearly
    from 0 to given `learning_rate` during warmup periods and keeps learning
    rate a constant.

    Args:
        learning_rate (float): The base learning rate. It is a python float
            number.
        warmup (int): The number of steps for warmup.
        last_epoch (int, optional): The index of last epoch. It can be set to
            restart training. If None, it means initial learning rate. 
            Default: -1.

    Examples:
        
        .. code-block:: python

            from paddlenlp.transformers import ConstSchedulerWithWarmup
            lr, warmup_steps = 0.1, 100
            lr_scheduler = ConstSchedulerWithWarmup(lr, warmup_steps)

    """

    def __init__(self,
                 learning_rate,
                 warmup,
                 total_steps=None,
                 last_epoch=-1,
                 verbose=False):
        warmup_steps = warmup if is_integer(warmup) else int(
            math.floor(warmup * total_steps))
        if is_integer(warmup):
            warmup_steps = warmup
        elif total_steps:
            warmup_steps = int(math.floor(warmup * total_steps))
        else:
            raise ValueError(
                "Please provide total steps if `warmup` is a float number , or provide integer for argument `warmup`."
            )

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1.0, warmup_steps))
            return 1.0

        super(ConstSchedulerWithWarmup, self).__init__(learning_rate, lr_lambda,
                                                       last_epoch, verbose)


class CosineDecayWithWarmUp(LambdaDecay):
    """
    Create a learning rate scheduler, which increases learning rate linearly
    from 0 to given `learning_rate`, after this warmup period learning rate
    would be decreased following the values of the cosine function. If
    `with_hard_restarts` is True, the cosine function could have serveral hard
    restarts.

    Args:
        learning_rate (float): The base learning rate. It is a python float
            number.
        total_steps (int): The number of training steps.
        warmup (int|float): If int, it means the number of steps for warmup.
            If float, it means the proportion of warmup in total training steps.
        with_hard_restarts (bool) Whether cosine function has several hard
            restarts. Default: False.
        num_cycles (int|float optional): If `with_hard_restarts` is False, it
            means the number of waves in cosine scheduler and should be an
            integer number and defaults to 1. If `with_hard_restarts` is True,
            it means the number of hard restarts to use and should be a float
            number and defaults to be 0.5. Default: None.
        last_epoch (int, optional): The index of last epoch. It can be set to
            restart training. If None, it means initial learning rate. 
            Default: -1.

    Examples:
        
        .. code-block:: python

            from paddlenlp.transformers import CosineDecayWithWarmUp
            lr, warmup_steps, max_steps = 0.1, 100, 1000
            lr_scheduler = CosineDecayWithWarmUp(lr, max_steps, warmup_steps)

    """

    def __init__(self,
                 learning_rate,
                 total_steps,
                 warmup,
                 with_hard_restarts=False,
                 num_cycles=None,
                 last_epoch=-1,
                 verbose=False):
        warmup_steps = warmup if is_integer(warmup) else int(
            math.floor(warmup * total_steps))
        # Input check
        if num_cycles is not None:
            assert not with_hard_restarts and isinstance(num_cycles, int) or with_hard_restarts and isinstance(num_cycles, float), \
            "`num_circles` should be and integer while `with_hard_restarts` is False, an float while `with_hard_restarts` is True."

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            if num_cycles is None:
                num_cycles = 1 if not with_hard_restarts else 0.5

            progress = float(current_step - warmup_steps) / float(
                max(1, total_steps - warmup_steps))

            if with_hard_restarts:
                if progress >= 1.0:
                    return 0.0
                return max(0.0, 0.5 * (1.0 + math.cos(math.pi * (
                    (float(num_cycles) * progress) % 1.0))))

            return max(0.0, 0.5 * (
                1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

        super(CosineDecayWithWarmUp, self).__init__(learning_rate, lr_lambda,
                                                    last_epoch, verbose)


class PolyDecayWithWarmUp(LambdaDecay):
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
            restart training. If None, it means initial learning rate.
            Default: -1.

    Examples:
        
        .. code-block:: python

            from paddlenlp.transformers import PolyDecayWithWarmUp
            lr, lr_end, warmup_steps, max_steps = 0.1, 1e-6, 100, 1000
            lr_scheduler = PolyDecayWithWarmUp(lr, max_steps, warmup_steps, lr_end)

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
                return lr_end / lr_init  # it multiplies by lr_init equals to lr_end
            else:
                lr_range = lr_init - lr_end
                decay_steps = total_steps - warmup_steps
                pct_remaining = 1 - (current_step - warmup_steps) / decay_steps
                decay = lr_range * pct_remaining**power + lr_end
                return decay / lr_init  # it multiplies by lr_init equals to decay

        super(PolyDecayWithWarmUp, self).__init__(lr_init, lr_lambda,
                                                  last_epoch, verbose)
