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

from functools import reduce
import importlib
import os
import numpy as np
import math
import paddle
from collections import defaultdict

from ppcv.ops.models.base import ModelBaseOp

from ppcv.ops.base import create_operators
from ppcv.core.workspace import register
from ppcv.ops.models.ocr.ocr_db_detection.preprocess import RGB2BGR
from .preprocess import *
from .postprocess import *


@register
class OcrCrnnRecOp(ModelBaseOp):
    def __init__(self, model_cfg, env_cfg):
        super(OcrCrnnRecOp, self).__init__(model_cfg, env_cfg)
        mod = importlib.import_module(__name__)
        self.preprocessor = create_operators(model_cfg["PreProcess"], mod)
        self.postprocessor = create_operators(model_cfg["PostProcess"], mod)
        self.batch_size = model_cfg["batch_size"]
        self.rec_image_shape = list(model_cfg["PreProcess"][-1].values())[0][
            "rec_image_shape"]

    @classmethod
    def get_output_keys(cls):
        return ["rec_text", "rec_score"]

    def preprocess(self, inputs):
        outputs = inputs
        for ops in self.preprocessor:
            outputs = ops(outputs)
        return outputs

    def postprocess(self, result):
        outputs = result
        for idx, ops in enumerate(self.postprocessor):
            if idx == len(self.postprocessor) - 1:
                outputs = ops(outputs, self.output_keys)
            else:
                outputs = ops(outputs)
        return outputs

    def infer(self, image_list):
        width_list = [float(img.shape[1]) / img.shape[0] for img in image_list]
        indices = np.argsort(np.array(width_list))

        inputs = []
        results = [None] * len(image_list)
        for beg_img_no in range(0, len(image_list), self.batch_size):
            end_img_no = min(len(image_list), beg_img_no + self.batch_size)
            imgC, imgH, imgW = self.rec_image_shape
            max_wh_ratio = imgW / imgH

            norm_img_batch = []
            for ino in range(beg_img_no, end_img_no):
                h, w = image_list[indices[ino]].shape[0:2]
                wh_ratio = w * 1.0 / h
                max_wh_ratio = max(max_wh_ratio, wh_ratio)

            for ino in range(beg_img_no, end_img_no):
                norm_img = self.preprocess({
                    'image': image_list[indices[ino]],
                    'max_wh_ratio': max_wh_ratio
                })['image']
                norm_img = norm_img[np.newaxis, :]
                norm_img_batch.append(norm_img)

            norm_img_batch = np.concatenate(norm_img_batch, axis=0)

            # model inference
            result = self.predictor.run(norm_img_batch)
            # postprocess
            result = self.postprocess(result)

            for rno in range(len(result)):
                results[indices[beg_img_no + rno]] = result[rno]
        return results

    def __call__(self, inputs):
        """
        step1: parser inputs
        step2: run
        step3: merge results
        input: a list of dict
        """
        key = self.input_keys[0]
        is_list = False
        if isinstance(inputs[0][key], (list, tuple)):
            inputs = [input[key] for input in inputs]
            is_list = True
        else:
            inputs = [[input[key]] for input in inputs]
        # expand a dim to adjust [[image,iamge],[image,image]] format
        expand_dim = False
        if isinstance(inputs[0][0], np.ndarray):
            inputs = [inputs]
            expand_dim = True
        pipe_outputs = []
        for i, images in enumerate(inputs):
            sub_index_list = [len(input) for input in images]
            images = reduce(lambda x, y: x.extend(y) or x, images)

            # step2: run
            outputs = self.infer(images)
            # step3: merge
            curr_offsef_id = 0
            results = []
            for idx in range(len(sub_index_list)):
                sub_start_idx = curr_offsef_id
                sub_end_idx = curr_offsef_id + sub_index_list[idx]
                output = outputs[sub_start_idx:sub_end_idx]
                if len(output) > 0:
                    output = {k: [o[k] for o in output] for k in output[0]}
                    if is_list is not True:
                        output = {k: output[k][0] for k in output}
                else:
                    output = {self.output_keys[0]: [], self.output_keys[1]: []}
                results.append(output)

                curr_offsef_id = sub_end_idx
            pipe_outputs.append(results)
        if expand_dim:
            pipe_outputs = pipe_outputs[0]
        else:
            outputs = []
            for pipe_output in pipe_outputs:
                d = defaultdict(list)
                for item in pipe_output:
                    for k in self.output_keys:
                        d[k].append(item[k])
                outputs.append(d)
            pipe_outputs = outputs
        return pipe_outputs
