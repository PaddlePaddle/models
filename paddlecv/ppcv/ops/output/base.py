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
import numpy as np
import math
import glob
import paddle
import cv2
from collections import defaultdict

from ppcv.ops.base import BaseOp


class OutputBaseOp(BaseOp):
    def __init__(self, model_cfg, env_cfg):
        super(OutputBaseOp, self).__init__(model_cfg, env_cfg)
        self.output_dir = self.env_cfg.get('output_dir', 'output')
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
        self.save_img = self.env_cfg.get('save_img', False)
        self.save_res = self.env_cfg.get('save_res', False)
        self.return_res = self.env_cfg.get('return_res', False)
        self.print_res = self.env_cfg.get('print_res', False)

    @classmethod
    def type(self):
        return 'OUTPUT'

    def __call__(self, inputs):
        return
