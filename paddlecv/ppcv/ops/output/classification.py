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
import json
from collections import defaultdict
from .base import OutputBaseOp
from ppcv.utils.logger import setup_logger
from ppcv.core.workspace import register

logger = setup_logger('ClasOutput')


@register
class ClasOutput(OutputBaseOp):
    def __init__(self, model_cfg, env_cfg):
        super(ClasOutput, self).__init__(model_cfg, env_cfg)

    def __call__(self, inputs):
        total_res = []
        for res in inputs:
            fn, image, class_ids, scores, label_names = res.values()
            image = res.pop('input.image')
            if self.frame_id != -1:
                res.update({'frame_id': frame_id})
            logger.info(res)
            if self.save_img:
                image = image[:, :, ::-1]
                file_name = os.path.split(fn)[-1]
                out_path = os.path.join(self.output_dir, file_name)
                logger.info('Save output image to {}'.format(out_path))
                cv2.imwrite(out_path, image)
            if self.save_res or self.return_res:
                total_res.append(res)
        if self.save_res:
            res_file_name = 'clas_output.json'
            out_path = os.path.join(self.output_dir, res_file_name)
            logger.info('Save output result to {}'.format(out_path))
            with open(out_path, 'w') as f:
                json.dump(total_res, f)
        if self.return_res:
            return total_res
        return
