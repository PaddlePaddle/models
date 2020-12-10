# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved
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

from __future__ import print_function

import math
from paddle.fluid.dygraph.learning_rate_scheduler import LearningRateDecay


class LinearLrWarmup(LearningRateDecay):
    """
    This operator use the linear learning rate warm up strategy to adjust the learning rate preliminarily before the normal learning rate scheduling.
    For more information, please refer to `Bag of Tricks for Image Classification with Convolutional Neural Networks <https://arxiv.org/abs/1812.01187>`_
    
    When global_step < warmup_steps, learning rate is updated as:
    
    .. code-block:: text
    
            linear_step = end_lr - start_lr
            lr = start_lr + linear_step * (global_step / warmup_steps)
    
    where start_lr is the initial learning rate, and end_lr is the final learning rate;
    
    When global_step >= warmup_steps, learning rate is updated as:
    
    .. code-block:: text
    
            lr = learning_rate
    
    where lr is the learning_rate after warm-up.
    
    Args:
        learning_rate (Variable|float): Learning_rate after warm-up, it could be 1D-Tensor or single value with the data type of float32.
        warmup_steps (int): Steps for warm up.
        start_lr (float): Initial learning rate of warm up.
        end_lr (float): Final learning rate of warm up.
        begin(int, optional): The begin step. The initial value of global_step described above. The default value is 0.
        step(int, optional): The step size used to calculate the new global_step in the description above.
            The default value is 1.
        dtype(str, optional): The data type used to create the learning rate variable. The data type can be set as
            'float32', 'float64'. The default value is 'float32'.
    
    Returns:
        Variable: Warm-up learning rate with the same data type as learning_rate.
    
    
    Examples:
    
    .. code-block:: python
    
        import paddle.fluid as fluid
    
        learning_rate = 0.1 
        warmup_steps = 50
        start_lr = 1. / 3.
        end_lr = 0.1

        with fluid.dygraph.guard(): 
            lr_decay = fluid.dygraph.LinearLrWarmup( learning_rate, warmup_steps, start_lr, end_lr)
    
       
    """

    def __init__(self,
                 learning_rate,
                 warmup_steps,
                 start_lr,
                 end_lr,
                 begin=1,
                 step=1,
                 dtype='float32'):
        super(LinearLrWarmup, self).__init__(begin, step, dtype)
        type_check = isinstance(learning_rate, float) or isinstance(
            learning_rate, int) or isinstance(learning_rate, LearningRateDecay)
        if not type_check:
            raise TypeError(
                "the type of learning_rate should be [int, float or LearningRateDecay], the current type is {}".
                format(learning_rate))
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        assert end_lr > start_lr, "end_lr {} must be greater than start_lr {}".format(
            end_lr, start_lr)
        self.lr_ratio_before_warmup = (
            float(end_lr) - float(start_lr)) / float(warmup_steps)
        self.start_lr = start_lr

    def step(self):
        base_lr = self.learning_rate
        if isinstance(self.learning_rate, LearningRateDecay):
            base_lr = base_lr()

        if self.step_num < self.warmup_steps:
            return self.start_lr + self.lr_ratio_before_warmup * self.step_num
        else:
            return base_lr
