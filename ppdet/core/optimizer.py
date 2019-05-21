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

import paddle.fluid.optimizer as optimizer
import paddle.fluid.layers as layers
import paddle.fluid.regularizer as regularizer

__all__ = ['OptimizerBuilder']


class OptimizerBuilder():
    """ Optimizer Builder
    Build optimizer handle base on given config AttrDict
    
    Args:
        cfg (AttrDict): config dict.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.lr = None

    def get_optimizer(self):
        return self._build_optimizer()

    def get_bn_regularizer(self):
        reg_cfg = self.cfg.WEIGHT_DECAY
        reg_func = getattr(regularizer, reg_cfg.TYPE + "Decay")
        return reg_func(reg_cfg.BN_FACTOR)

    def _build_optimizer(self):
        regularization = self._build_regularizer()
        learning_rate = self._build_learning_rate()

        opt_params = dict()
        for k, v in self.cfg.items():
            if k == 'TYPE':
                opt_func = getattr(optimizer, v)
            elif not isinstance(v, dict):
                opt_params.update({k.lower(): v})

        # parse warmup
        warmup_cfg = self.cfg.LR_WARMUP
        if self.cfg.LR_WARMUP.WARMUP:
            warmup_steps = warmup_cfg.WARMUP_STEPS
            end_lr = warmup_cfg.WARMUP_END_LR
            start_lr = warmup_cfg.WARMUP_INIT_FACTOR * \
                       warmup_cfg.WARMUP_END_LR
            learning_rate = layers.linear_lr_warmup(
                learning_rate=learning_rate,
                warmup_steps=warmup_steps,
                start_lr=start_lr,
                end_lr=end_lr)
        self.lr = learning_rate
        return opt_func(
            regularization=regularization,
            learning_rate=learning_rate,
            **opt_params)

    def _build_regularizer(self):
        reg_cfg = self.cfg.WEIGHT_DECAY
        reg_func = getattr(regularizer, reg_cfg.TYPE + "Decay")
        return reg_func(reg_cfg.FACTOR)

    def _build_learning_rate(self):
        policy, params = self._parse_lr_decay_cfg()
        decay_func = getattr(layers, policy)
        return decay_func(**params)

    def _parse_lr_decay_cfg(self):
        lr_cfg = self.cfg.LR_DECAY
        policy = lr_cfg.POLICY
        # parse decay params
        params = dict()
        for k, v in lr_cfg.items():
            if k != 'POLICY':
                params.update({k.lower(): v})
        return policy, params
