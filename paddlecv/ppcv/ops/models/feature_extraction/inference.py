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

import importlib
from functools import reduce
import os
import numpy as np
import math
import paddle
from ..base import ModelBaseOp

from ppcv.ops.base import create_operators
from ppcv.core.workspace import register

from .preprocess import *
from .postprocess import *


@register
class FeatureExtractionOp(ModelBaseOp):
    def __init__(self, model_cfg, env_cfg):
        super().__init__(model_cfg, env_cfg)
        mod = importlib.import_module(__name__)
        self.preprocessor = create_operators(model_cfg["PreProcess"], mod)
        self.postprocessor = create_operators(model_cfg["PostProcess"], mod)

    @classmethod
    def get_output_keys(cls):
        return ["dt_bboxes", "feature", "rec_score", "rec_doc"]

    def preprocess(self, inputs):
        outputs = inputs
        for ops in self.preprocessor:
            outputs = ops(outputs)
        return outputs

    def postprocess(self, output_list, bbox_list):
        assert len(output_list) == len(bbox_list)
        if len(output_list) == 0:
            return {k: None for k in self.output_keys}
        output_dict = {
            self.output_keys[0]: bbox_list,
            self.output_keys[1]: output_list
        }
        for idx, ops in enumerate(self.postprocessor):
            output_dict = ops(output_dict, self.output_keys)
        return output_dict

    def infer_img(self, input):
        # predict the full input image
        img = input[self.input_keys[0]]
        h, w = img.shape[:2]
        image_list = [img]
        bbox_list = [[0, 0, w, h]]
        # for cropped image from object detection
        if len(self.input_keys) == 3:
            image_list.extend(input[self.input_keys[1]])
            bbox_list.extend(input[self.input_keys[2]])

        batch_loop_cnt = math.ceil(float(len(image_list)) / self.batch_size)
        output_list = []
        for i in range(batch_loop_cnt):
            start_index = i * self.batch_size
            end_index = min((i + 1) * self.batch_size, len(image_list))
            batch_image_list = image_list[start_index:end_index]
            # preprocess
            inputs = [self.preprocess(img) for img in batch_image_list]
            inputs = np.concatenate(inputs, axis=0)
            # model inference
            output = self.predictor.run(inputs)[0]
            output_list.extend(output)
        # postprocess
        return self.postprocess(output_list, bbox_list)

    def __call__(self, inputs):
        outputs = []
        for input in inputs:
            output = self.infer_img(input)
            outputs.append(output)
        return outputs
