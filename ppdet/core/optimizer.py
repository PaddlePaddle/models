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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

import paddle.fluid.optimizer as optimizer
import paddle.fluid.layers as layers
import paddle.fluid.regularizer as regularizer

__all__ = ['OptimizerBuilder']

logger = logging.getLogger(__name__)


class OptimizerBuilder():
    """ Optimizer Builder
    Build optimizer handle base on given config AttrDict
    
    Args:
        cfg (AttrDict): config dict.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.learning_rate = None
        self.optimizer = None

    def get_optimizer(self):
        return self.optimizer or self._build_optimizer()

    def get_lr(self):
        """
        Get learning variable.
        """
        if self.learning_rate is None:
            raise ValueError("learning_rate is not initialized.")
        return self.learning_rate

    def _build_optimizer(self):
        self.regularization = self._build_regularization()
        self.learning_rate = self._build_learning_rate()

        opt_params = dict()
        for k, v in self.cfg.items():
            if k == 'TYPE':
                opt_func = getattr(optimizer, v)
            elif not isinstance(v, dict):
                opt_params.update({k.lower(): v})

        self.optimizer =  opt_func(regularization=self.regularization,
                                   learning_rate=self.learning_rate,
                                   **opt_params)
        return self.optimizer

    def _build_regularization(self):
        reg_cfg = self.cfg.WEIGHT_DECAY
        reg_func = getattr(regularizer, reg_cfg.TYPE + "Decay")
        return reg_func(reg_cfg.FACTOR)

    def _build_learning_rate(self):
        # parse and perform learning rate decay
        policy, params = self._parse_lr_decay_cfg()
        decay_func = getattr(layers, policy)
        learning_rate = decay_func(**params)

        # parse and perform warmup
        warmup_cfg = self.cfg.LR_WARMUP
        if self.cfg.LR_WARMUP.WARMUP:
            warmup_steps = warmup_cfg.WARMUP_STEPS
            end_lr = warmup_cfg.WARMUP_END_LR
            start_lr = warmup_cfg.WARMUP_INIT_FACTOR * \
                       warmup_cfg.WARMUP_END_LR
            logger.info("Learning rate warm up from {:4f} to {:4f} "
                        "in {} steps.".format(start_lr, end_lr, 
                                             warmup_steps))
            learning_rate = layers.linear_lr_warmup(
                learning_rate=learning_rate,
                warmup_steps=warmup_steps,
                start_lr=start_lr,
                end_lr=end_lr)

        self.lr = learning_rate
        return learning_rate

        self.lr = learning_rate
        return learning_rate

    def _parse_lr_decay_cfg(self):
        lr_cfg = self.cfg.LR_DECAY
        policy = lr_cfg.POLICY
        # parse decay params
        params = dict()
        for k, v in lr_cfg.items():
            if k != 'POLICY':
                params.update({k.lower(): v})
        return policy, params
