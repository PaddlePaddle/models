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

from ppcv.ops.base import create_operators, BaseOp
from ppcv.core.workspace import register
from ppcv.utils.utility import check_install

from .postprocess import *


@register
class SentimentAnalysisOp(BaseOp):
    def __init__(self, model_cfg, env_cfg):
        super(SentimentAnalysisOp, self).__init__(model_cfg, env_cfg)
        mod = importlib.import_module(__name__)
        env_cfg["batch_size"] = model_cfg.get("batch_size", 1)
        self.batch_size = env_cfg["batch_size"]
        self.name = model_cfg["name"]
        self.frame = -1
        keys = self.get_output_keys()
        self.output_keys = [self.name + '.' + key for key in keys]

        self.postprocessor = create_operators(model_cfg["PostProcess"], mod)
        self._init_task(model_cfg)

    def _init_task(self, model_cfg):
        check_install('paddlenlp', 'paddlenlp')
        from paddlenlp import Taskflow
        task = model_cfg.get('task', 'sentiment_analysis')
        self.nlp = Taskflow(task)

    @classmethod
    def get_output_keys(cls):
        return ["label"]

    def postprocess(self, inputs, result):
        outputs = result
        for idx, ops in enumerate(self.postprocessor):
            if idx == len(self.postprocessor) - 1:
                outputs = ops(outputs, self.output_keys)
            else:
                outputs = ops(outputs)
        return outputs

    @classmethod
    def type(self):
        return 'MODEL'

    def infer(self, image_list):
        inputs = []
        batch_loop_cnt = math.ceil(float(len(image_list)) / self.batch_size)
        results = []
        for i in range(batch_loop_cnt):
            start_index = i * self.batch_size
            end_index = min((i + 1) * self.batch_size, len(image_list))
            batch_image_list = image_list[start_index:end_index]
            # preprocess
            # model inference
            result = self.nlp(batch_image_list)
            # postprocess
            result = self.postprocess(inputs, result)
            results.extend(result)
        # results = self.merge_batch_result(results)
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

        # step2: run
        outputs = self.infer(inputs)

        # step3: merge
        curr_offsef_id = 0
        pipe_outputs = []
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


@register
class InformationExtractionOp(SentimentAnalysisOp):
    def __init__(self, model_cfg, env_cfg):
        super(InformationExtractionOp, self).__init__(model_cfg, env_cfg)
        self._init_task(model_cfg)

    def _init_task(self, model_cfg):
        check_install('paddlenlp', 'paddlenlp')
        from paddlenlp import Taskflow
        task = model_cfg.get('task', 'information_extraction')
        schema = model_cfg.get('schema', ['时间', '地点', '人物'])
        self.nlp = Taskflow(task, schema=schema)

    @classmethod
    def get_output_keys(cls):
        return ["text", "type"]
