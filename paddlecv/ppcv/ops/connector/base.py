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

import cv2
import numpy as np

from ppcv.ops.base import BaseOp


class ConnectorBaseOp(BaseOp):
    def __init__(self, model_cfg, env_cfg=None):
        super(ConnectorBaseOp, self).__init__(model_cfg, env_cfg)
        self.name = model_cfg["name"]
        keys = self.get_output_keys()
        self.output_keys = [self.name + '.' + key for key in keys]

    @classmethod
    def type(self):
        return 'CONNECTOR'
