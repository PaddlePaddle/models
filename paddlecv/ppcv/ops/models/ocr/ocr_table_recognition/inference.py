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
import math

from ppcv.ops.base import create_operators
from ppcv.core.workspace import register
from ppcv.ops.models.base import ModelBaseOp

from ppcv.ops.models.ocr.ocr_db_detection.preprocess import NormalizeImage, ToCHWImage, KeepKeys, ExpandDim, RGB2BGR
from ppcv.ops.models.ocr.ocr_table_recognition.preprocess import *
from ppcv.ops.models.ocr.ocr_table_recognition.postprocess import *


@register
class PPStructureTableStructureOp(ModelBaseOp):
    def __init__(self, model_cfg, env_cfg):
        super(PPStructureTableStructureOp, self).__init__(model_cfg, env_cfg)
        mod = importlib.import_module(__name__)
        self.preprocessor = create_operators(model_cfg["PreProcess"], mod)
        self.postprocessor = create_operators(model_cfg["PostProcess"], mod)
        self.batch_size = model_cfg["batch_size"]

    @classmethod
    def get_output_keys(cls):
        return ["structures", "dt_bboxes", "scores"]

    def preprocess(self, inputs):
        outputs = inputs
        for ops in self.preprocessor:
            outputs = ops(outputs)
        return outputs

    def postprocess(self, result, shape_list):
        outputs = result
        for idx, ops in enumerate(self.postprocessor):
            if idx == len(self.postprocessor) - 1:
                outputs = ops(outputs, shape_list, self.output_keys)
            else:
                outputs = ops(outputs)
        return outputs

    def infer(self, image_list):
        batch_loop_cnt = math.ceil(float(len(image_list)) / self.batch_size)
        results = []
        for i in range(batch_loop_cnt):
            start_index = i * self.batch_size
            end_index = min((i + 1) * self.batch_size, len(image_list))
            batch_image_list = image_list[start_index:end_index]
            # preprocess
            inputs = [
                self.preprocess({
                    'image': img
                }) for img in batch_image_list
            ]
            shape_list = np.stack([x['shape'] for x in inputs])
            inputs = np.concatenate([x['image'] for x in inputs], axis=0)
            # model inference
            result = self.predictor.run(inputs)
            # postprocess
            result = self.postprocess(result, shape_list)
            results.extend(result)
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
        sub_index_list = [len(input) for input in inputs]
        inputs = reduce(lambda x, y: x.extend(y) or x, inputs)

        pipe_outputs = []
        if len(inputs) == 0:
            pipe_outputs.append({
                self.output_keys[0]: [],
                self.output_keys[1]: [],
                self.output_keys[2]: [],
            })
            return pipe_outputs
        # step2: run
        outputs = self.infer(inputs)
        # step3: merge
        curr_offsef_id = 0
        for idx in range(len(sub_index_list)):
            sub_start_idx = curr_offsef_id
            sub_end_idx = curr_offsef_id + sub_index_list[idx]
            output = outputs[sub_start_idx:sub_end_idx]
            output = {k: [o[k] for o in output] for k in output[0]}
            if is_list is not True:
                output = {k: output[k][0] for k in output}
            pipe_outputs.append(output)

            curr_offsef_id = sub_end_idx
        return pipe_outputs
