# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" 
Copy-paste from PaddleSeg with minor modifications.
https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.1/paddleseg/cvlibs/config.py
"""

import codecs
import os
from typing import Any, Dict, Generic

import paddle
import yaml

from smoke.cvlibs import manager
from smoke.utils import logger

class Config(object):
    '''
    Training configuration parsing. The only yaml/yml file is supported.

    The following hyper-parameters are available in the config file:
        batch_size: The number of samples per gpu.
        iters: The total training steps.
        train_dataset: A training data config including type/data_root/transforms/mode.
            For data type, please refer to paddleseg.datasets.
            For specific transforms, please refer to paddleseg.transforms.transforms.
        val_dataset: A validation data config including type/data_root/transforms/mode.
        optimizer: A optimizer config, but currently PaddleSeg only supports sgd with momentum in config file.
            In addition, weight_decay could be set as a regularization.
        learning_rate: A learning rate config. If decay is configured, learning _rate value is the starting learning rate,
             where only poly decay is supported using the config file. In addition, decay power and end_lr are tuned experimentally.
        loss: A loss config. Multi-loss config is available. The loss type order is consistent with the seg model outputs,
            where the coef term indicates the weight of corresponding loss. Note that the number of coef must be the same as the number of
            model outputs, and there could be only one loss type if using the same loss type among the outputs, otherwise the number of
            loss type must be consistent with coef.
        model: A model config including type/backbone and model-dependent arguments.
            For model type, please refer to paddleseg.models.
            For backbone, please refer to paddleseg.models.backbones.

    Args:
        path (str) : The path of config file, supports yaml format only.

    Examples:

        from paddleseg.cvlibs.config import Config

        # Create a cfg object with yaml file path.
        cfg = Config(yaml_cfg_path)

        # Parsing the argument when its property is used.
        train_dataset = cfg.train_dataset

        # the argument of model should be parsed after dataset,
        # since the model builder uses some properties in dataset.
        model = cfg.model
        ...
    '''

    def __init__(self,
                 path: str,
                 learning_rate: float = None,
                 batch_size: int = None,
                 iters: int = None):
        if not path:
            raise ValueError('Please specify the configuration file path.')

        if not os.path.exists(path):
            raise FileNotFoundError('File {} does not exist'.format(path))

        self._model = None

        if path.endswith('yml') or path.endswith('yaml'):
            self.dic = self._parse_from_yaml(path)
        else:
            raise RuntimeError('Config file should in yaml format!')

        self.update(
            learning_rate=learning_rate, batch_size=batch_size, iters=iters)

    def _update_dic(self, dic, base_dic):
        """
        Update config from dic based base_dic
        """
        base_dic = base_dic.copy()
        for key, val in dic.items():
            if isinstance(val, dict) and key in base_dic:
                base_dic[key] = self._update_dic(val, base_dic[key])
            else:
                base_dic[key] = val
        dic = base_dic
        return dic

    def _parse_from_yaml(self, path: str):
        '''Parse a yaml file and build config'''
        with codecs.open(path, 'r', 'utf-8') as file:
            dic = yaml.load(file, Loader=yaml.FullLoader)

        if '_base_' in dic:
            cfg_dir = os.path.dirname(path)
            base_path = dic.pop('_base_')
            base_path = os.path.join(cfg_dir, base_path)
            base_dic = self._parse_from_yaml(base_path)
            dic = self._update_dic(dic, base_dic)
        return dic

    def update(self,
               learning_rate: float = None,
               batch_size: int = None,
               iters: int = None):
        '''Update config'''
        if learning_rate:
            self.dic['lr_scheduler']['learning_rate'] = learning_rate

        if batch_size:
            self.dic['batch_size'] = batch_size

        if iters:
            self.dic['iters'] = iters

    @property
    def batch_size(self):
        return self.dic.get('batch_size', 1)

    @property
    def iters(self):
        iters = self.dic.get('iters')
        if not iters:
            raise RuntimeError('No iters specified in the configuration file.')
        return iters


    @property
    def train_dataset(self):
        train_dataset_cfg = self.dic.get('train_dataset', {})
        if not train_dataset_cfg:
            return None
        return self._load_object(train_dataset_cfg)
    
    @property
    def val_dataset(self):
        val_dataset_cfg = self.dic.get('val_dataset', {})
        if not val_dataset_cfg:
            return None
        return self._load_object(val_dataset_cfg)

    @property
    def model(self):
        model_cfg = self.dic.get('model').copy()
        if not model_cfg:
            raise RuntimeError('No model specified in the configuration file.')

        if not self._model:
            self._model = self._load_object(model_cfg)
        return self._model

    @property
    def lr_scheduler(self) -> paddle.optimizer.lr.LRScheduler:
        if 'lr_scheduler' not in self.dic:
            raise RuntimeError(
                'No `lr_scheduler` specified in the configuration file.')
        params = self.dic.get('lr_scheduler').copy()
        if 'type' not in params.keys():
            if "learning_rate" in params.keys():
                logger.warning(''' No decay config! The fixed learning rate will be used''')
                return params["learning_rate"]
            else:
                raise RuntimeError(
                '`lr_scheduler` is not set properlly in the configuration file.')
        
        lr_type = params.pop('type')

        return getattr(paddle.optimizer.lr, lr_type)(**params)

    @property
    def optimizer(self):
        if 'lr_scheduler' in self.dic:
            lr = self.lr_scheduler
        else:
            lr = self.learning_rate
        args = self.dic.get('optimizer', {}).copy()
        optimizer_type = args.pop('type')

        return getattr(paddle.optimizer, optimizer_type)(lr, parameters=self.model.parameters(), **args)

    @property
    def loss(self):
        loss_cfg = self.dic.get('loss', {}).copy()
        if not loss_cfg:
            return None
        return self._load_object(loss_cfg)

    def _load_component(self, com_name):
        com_list = [
            manager.MODELS, manager.BACKBONES, manager.DATASETS,
            manager.TRANSFORMS, manager.LOSSES, manager.HEADS,
            manager.POSTPROCESSORS
        ]

        for com in com_list:
            if com_name in com.components_dict:
                return com[com_name]
        else:
            raise RuntimeError(
                'The specified component was not found {}.'.format(com_name))

    def _load_object(self, cfg):
        cfg = cfg.copy()
        if 'type' not in cfg:
            raise RuntimeError('No object information in {}.'.format(cfg))

        component = self._load_component(cfg.pop('type'))

        params = {}
        for key, val in cfg.items():
            if self._is_meta_type(val):
                params[key] = self._load_object(val)
            elif isinstance(val, list):
                params[key] = [
                    self._load_object(item)
                    if self._is_meta_type(item) else item for item in val
                ]
            else:
                params[key] = val

        return component(**params)


    def _is_meta_type(self, item):
        return isinstance(item, dict) and 'type' in item

    def __str__(self):
        return yaml.dump(self.dic)
