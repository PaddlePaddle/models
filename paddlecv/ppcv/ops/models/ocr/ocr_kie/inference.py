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

from ppcv.ops.models.classification.preprocess import ResizeImage
from ppcv.ops.models.ocr.ocr_db_detection.preprocess import NormalizeImage, ToCHWImage, KeepKeys
from ppcv.ops.models.ocr.ocr_kie.preprocess import *
from ppcv.ops.models.ocr.ocr_kie.postprocess import *


@register
class PPStructureKieSerOp(ModelBaseOp):
    def __init__(self, model_cfg, env_cfg):
        super(PPStructureKieSerOp, self).__init__(model_cfg, env_cfg)
        mod = importlib.import_module(__name__)
        self.preprocessor = create_operators(model_cfg["PreProcess"], mod)
        self.postprocessor = create_operators(model_cfg["PostProcess"], mod)
        self.batch_size = model_cfg["batch_size"]
        self.use_visual_backbone = model_cfg.get('use_visual_backbone', False)

    @classmethod
    def get_output_keys(cls):
        return ["pred_id", "pred", "dt_polys", "rec_text", "inputs"]

    def preprocess(self, inputs):
        outputs = inputs
        for ops in self.preprocessor:
            outputs = ops(outputs)
        return outputs

    def postprocess(self, result, segment_offset_ids, ocr_infos):
        outputs = result
        for idx, ops in enumerate(self.postprocessor):
            if idx == len(self.postprocessor) - 1:
                outputs = ops(outputs,
                              segment_offset_ids=segment_offset_ids,
                              ocr_infos=ocr_infos)
            else:
                outputs = ops(outputs)
        return outputs

    def infer(self, data_list):
        batch_loop_cnt = math.ceil(float(len(data_list)) / self.batch_size)
        results = []
        ser_inputs = []
        for i in range(batch_loop_cnt):
            start_index = i * self.batch_size
            end_index = min((i + 1) * self.batch_size, len(data_list))
            batch_data_list = data_list[start_index:end_index]
            # preprocess
            inputs = [
                self.preprocess({
                    'image': data[self.input_keys[0]],
                    'ocr': {
                        'dt_polys': data[self.input_keys[1]],
                        'rec_text': data[self.input_keys[2]]
                    }
                }) for data in batch_data_list
            ]
            ser_inputs.extend(inputs)
            # concat to batch
            model_inputs = []
            for i in range(len(inputs[0])):
                x = [x[i] for x in inputs]
                if isinstance(x[0], np.ndarray):
                    x = np.stack(x)
                model_inputs.append(x)
            # model inference
            if self.use_visual_backbone:
                result = self.predictor.run(model_inputs[:5])
            else:
                result = self.predictor.run(model_inputs[:4])
            # postprocess
            result = self.postprocess(
                result[0],
                segment_offset_ids=model_inputs[6],
                ocr_infos=model_inputs[7])
            results.extend(result)
        return results, ser_inputs

    def __call__(self, inputs):
        """
        step1: parser inputs
        step2: run
        step3: merge results
        input: a list of dict
        """
        # step2: run
        outputs, ser_inputs = self.infer(inputs)
        # step3: merge
        pipe_outputs = []
        for output, ser_input in zip(outputs, ser_inputs):
            d = defaultdict(list)
            for res in output:
                d[self.output_keys[0]].append(res['pred_id'])
                d[self.output_keys[1]].append(res['pred'])
                d[self.output_keys[2]].append(res['points'])
                d[self.output_keys[3]].append(res['transcription'])
            d[self.output_keys[4]] = ser_input
            pipe_outputs.append(d)
        return pipe_outputs


@register
class PPStructureKieReOp(ModelBaseOp):
    def __init__(self, model_cfg, env_cfg):
        super(PPStructureKieReOp, self).__init__(model_cfg, env_cfg)
        mod = importlib.import_module(__name__)
        self.preprocessor = create_operators(model_cfg["PreProcess"], mod)
        self.postprocessor = create_operators(model_cfg["PostProcess"], mod)
        self.batch_size = model_cfg["batch_size"]
        self.use_visual_backbone = model_cfg.get('use_visual_backbone', False)

    @classmethod
    def get_output_keys(cls):
        return ["head", "tail"]

    def preprocess(self, inputs):
        outputs = inputs
        for ops in self.preprocessor:
            outputs = ops(outputs)
        return outputs

    def postprocess(self, result, **kwargs):
        outputs = result
        for idx, ops in enumerate(self.postprocessor):
            if idx == len(self.postprocessor) - 1:
                outputs = ops(outputs, **kwargs)
            else:
                outputs = ops(outputs)
        return outputs

    def infer(self, data_list):
        batch_loop_cnt = math.ceil(float(len(data_list)) / self.batch_size)
        results = []
        for i in range(batch_loop_cnt):
            start_index = i * self.batch_size
            end_index = min((i + 1) * self.batch_size, len(data_list))
            batch_data_list = data_list[start_index:end_index]
            # preprocess
            inputs = [
                self.preprocess({
                    'ser_inputs': data['ser.inputs'],
                    'ser_preds': data['ser.pred']
                }) for data in batch_data_list
            ]
            # concat to batch
            model_inputs = []
            for i in range(len(inputs[0])):
                x = [x[i] for x in inputs]
                if isinstance(x[0], np.ndarray):
                    x = np.stack(x)
                model_inputs.append(x)
            # model inference
            if not self.use_visual_backbone:
                model_inputs.pop(4)

            result = self.predictor.run(model_inputs[:-1])

            preds = dict(
                loss=result[1],
                pred_relations=result[2],
                hidden_states=result[0])

            # postprocess
            result = self.postprocess(
                preds,
                ser_results=batch_data_list,
                entity_idx_dict_batch=model_inputs[-1])
            results.extend(result)
        return results

    def __call__(self, inputs):
        """
        step1: parser inputs
        step2: run
        step3: merge results
        input: a list of dict
        """
        # step2: run
        outputs = self.infer(inputs)
        # step3: merge
        pipe_outputs = []
        for output in outputs:
            d = defaultdict(list)
            for res in output:
                d[self.output_keys[0]].append(res[0])
                d[self.output_keys[1]].append(res[1])
            pipe_outputs.append(d)
        return pipe_outputs
