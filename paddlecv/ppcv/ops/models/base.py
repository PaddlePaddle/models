# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved. 
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

import os
import importlib
import math
import numpy as np
import paddle
from paddle.inference import Config
from paddle.inference import create_predictor

from ppcv.ops.base import BaseOp
from ppcv.ops.predictor import PaddlePredictor
from ppcv.utils.download import get_model_path


class ModelBaseOp(BaseOp):
    """
    Base Operator, implement of prediction process
    Args
    """

    def __init__(self, model_cfg, env_cfg):
        super(ModelBaseOp, self).__init__(model_cfg, env_cfg)
        param_path = get_model_path(model_cfg['param_path'])
        model_path = get_model_path(model_cfg['model_path'])
        env_cfg["batch_size"] = model_cfg.get("batch_size", 1)
        delete_pass = model_cfg.get("delete_pass", [])
        self.batch_size = env_cfg["batch_size"]
        self.name = model_cfg["name"]
        self.frame = -1
        self.predictor = PaddlePredictor(param_path, model_path, env_cfg,
                                         delete_pass, self.name)
        self.input_names = self.predictor.get_input_names()

        keys = self.get_output_keys()
        self.output_keys = [self.name + '.' + key for key in keys]

    @classmethod
    def type(self):
        return 'MODEL'

    def preprocess(self, inputs):
        raise NotImplementedError

    def postprocess(self, inputs):
        raise NotImplementedError
