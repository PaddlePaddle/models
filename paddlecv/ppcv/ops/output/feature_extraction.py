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
import glob
import copy
import math
import json

import numpy as np
import paddle
import cv2

from collections import defaultdict
from .base import OutputBaseOp
from ppcv.utils.logger import setup_logger
from ppcv.core.workspace import register

logger = setup_logger('FeatureOutput')


@register
class FeatureOutput(OutputBaseOp):
    def __init__(self, model_cfg, env_cfg):
        super().__init__(model_cfg, env_cfg)

    def __call__(self, inputs):
        total_res = []
        for res in inputs:
            # TODO(gaotingquan): video input is not tested
            if self.frame_id != -1:
                res.update({'frame_id': frame_id})
            if self.print_res:
                msg = " ".join([f"{key}: {res[key]}" for key in res])
                logger.info(msg)
            if self.return_res:
                total_res.append(res)
        if self.return_res:
            return total_res
        return
