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
"""
Model base
"""

import paddle.fluid as fluid
from paddle.fluid.dygraph import parallel_helper


class ModelBase(fluid.dygraph.Layer):
    """
    Basic model wrapper for static graph and dygrpah.
    """
    _registry = dict()

    @classmethod
    def register(cls, name):
        ModelBase._registry[name] = cls
        return

    @staticmethod
    def by_name(name):
        return ModelBase._registry[name]

    @staticmethod
    def create(name_scope, hparams, *args, **kwargs):
        model_cls = ModelBase.by_name(hparams.model)
        return model_cls(name_scope, hparams, *args, **kwargs)

    @classmethod
    def add_cmdline_argument(cls, parser):
        """ Add cmdline argument. """
        group = parser.add_argument_group("Model")
        group.add_argument("--init_checkpoint", type=str, default=None)
        group.add_argument("--model", type=str, default="UnifiedTransformer",
                           choices=["UnifiedTransformer"])
        args, _ = parser.parse_known_args()
        model_cls = ModelBase.by_name(args.model)
        model_cls.add_cmdline_argument(group)
        return group

    def __init__(self, name_scope, hparams):
        super().__init__(name_scope)
        self.init_checkpoint = hparams.init_checkpoint
        return

    def __call__(self, *args, **kwargs):
        """ Re-implement __call__ function in dygraph mode. """
        if not self._built:
            self._build_once(*args, **kwargs)
            self._built = True

        outputs = self.forward(*args, **kwargs)
        return outputs

    def _build_once(self, inputs, *args, **kwargs):
        """
        Build only once.

        1. Initialize models's parameters.
        2. Boardcast parameters if in data parallel mode.
        3. Load saved parameters
        """
        # Initial parameters.
        self._create_parameters()

        if parallel_helper._is_data_parallel_mode():
            parallel_helper._broadcast_parameters(self._parameters.values())

        # Load persitables
        self._load_params()
        return

    def _create_parameters(self):
        """ Create model's paramters. """
        raise NotImplementedError

    def _load_params(self):
        """ Load saved paramters. """
        raise NotImplementedError

    def _forward(self, inputs, is_training):
        """ Real forward process of model in different mode(train/test). """
        raise NotImplementedError

    def _collect_metrics(self, inputs, outputs):
        """ Calculate loss function by using inputs and outputs. """
        raise NotImplementedError

    def _optimize(self, loss):
        """ Optimize loss function and update model. """
        raise NotImplementedError

    def _infer(self, inputs):
        """ Real inference process of model. """
        raise NotImplementedError

    def forward(self, inputs, is_training=False):
        """
        Forward process, include real forward, collect metrices and optimize(optional)

        @params : inputs : input data
        @type : dict of numpy.ndarray/int/float/...
        """
        if is_training:
            self.train()
        else:
            self.eval()

        outputs = self._forward(inputs, is_training)
        metrics = self._collect_metrics(inputs, outputs)
        loss = metrics["loss"]
        if is_training:
            self._optimize(loss)

        metrics = {k: v.numpy() for k, v in metrics.items()}
        return metrics

    def infer(self, inputs):
        """
        Inference process.

        @params : inputs : input data
        @type : dict of numpy.ndarray/int/float/...
        """
        if not self._built:
            self._build_once(inputs)
            self._built = True

        self.eval()
        results = self._infer(inputs)
        results = {name: results[name].numpy() for name in results}
        return results
