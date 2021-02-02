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

from paddle.optimizer.lr import LambdaDecay

__all__ = [
    'LinearSchedulerWithWarmup', 'ConstantSchedulerWithWarmup',
    'CosineSchedulerWithWarmup', 'CosineSchedulerWithWarmup',
    'CosineWithHardRestartsScheduleWithWarmup',
    'PolynomialDecaySchedulerWithWarmup'
]


class LinearSchedulerWithWarmup(LambdaDecay):
    """
    Create a learning rate scheduler, which increases learning rate linearly
    from 0 to given `learning_rate`, after this warmup period learning rate
    would be decreased linearly from the base learning rate to 0.

    Args:
        learning_rate (float): The base learning rate. It is a python float
            number.
        num_training_steps (int): The number of training steps.
        num_warmup_steps (int): The number of steps for warmup.
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
                 num_training_steps,
                 num_warmup_steps,
                 last_epoch=-1,
                 verbose=False):
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(0.0,
                       float(num_training_steps - current_step) /
                       float(max(1, num_training_steps - num_warmup_steps)))

        super(LinearSchedulerWithWarmup, self).__init__(
            learning_rate, lr_lambda, last_epoch, verbose)


class ConstantSchedulerWithWarmup(LambdaDecay):
    """
    Create a learning rate scheduler, which increases learning rate linearly
    from 0 to given `learning_rate` during warmup periods and keeps learning
    rate constant.

    Args:
        learning_rate (float): The base learning rate. It is a python float
            number.
        num_warmup_steps (int): The number of steps for warmup.
        last_epoch (int, optional): The index of last epoch. It can be set to
            resart training. If None, it means initial learning rate. 
            Default: -1.

    Examples:
        
        .. code-block:: python

            from paddlenlp.transformers import ConstantSchedulerWithWarmup
            lr, warmup_steps = 0.1, 100
            lr_scheduler = ConstantSchedulerWithWarmup(lr, warmup_steps)

    """

    def __init__(self,
                 learning_rate,
                 num_warmup_steps,
                 last_epoch=-1,
                 verbose=False):
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1.0, num_warmup_steps))
            return 1.0

        super(ConstantSchedulerWithWarmup, self).__init__(
            learning_rate, lr_lambda, last_epoch, verbose)


class CosineSchedulerWithWarmup(LambdaDecay):
    """
    Create a learning rate scheduler, which increases learning rate linearly
    from 0 to given `learning_rate`, after this warmup period learning rate
    would be decreased following the values of the cosine function.

    Args:
        learning_rate (float): The base learning rate. It is a python float
            number.
        num_training_steps (int): The number of training steps.
        num_warmup_steps (int): The number of steps for warmup.
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
                 num_training_steps,
                 num_warmup_steps,
                 num_cycles=0.5,
                 last_epoch=-1,
                 verbose=False):
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            progress = float(current_step - num_warmup_steps) / float(
                max(1, num_training_steps - num_warmup_steps))
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
        num_training_steps (int): The number of training steps.
        num_warmup_steps (int): The number of steps for warmup.
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
                 num_warmup_steps,
                 num_training_steps,
                 num_cycles=1,
                 last_epoch=-1,
                 verbose=False):
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            progress = float(current_step - num_warmup_steps) / float(
                max(1, num_training_steps - num_warmup_steps))
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
        num_training_steps (int): The number of training steps.
        num_warmup_steps (int): The number of steps for warmup.
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
                 num_training_steps,
                 num_warmup_steps,
                 lr_end=1e-7,
                 power=1.0,
                 last_epoch=-1,
                 verbose=False):
        lr_init = learning_rate
        assert lr_init > lr_end, f"`lr_end` must be be smaller than `learning_rate`. But `lr_end` is {lr_end} while `learning_rate` is {lr_init}."

        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            elif current_step > num_training_steps:
                return lr_end / lr_init  # which multiplies by lr_init equals to lr_end
            else:
                lr_range = lr_init - lr_end
                decay_steps = num_training_steps - num_warmup_steps
                pct_remaining = 1 - (current_step - num_warmup_steps
                                     ) / decay_steps
                decay = lr_range * pct_remaining**power + lr_end
                return decay / lr_init  # which multiplies by lr_init equals to decay

        super(PolynomialDecaySchedulerWithWarmup, self).__init__(
            lr_init, lr_lambda, last_epoch, verbose)
