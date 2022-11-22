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
import os
import importlib
import numpy as np
import math
import paddle
from ppcv.ops.base import create_operators
from ppcv.ops.models.base import ModelBaseOp
from ppcv.core.workspace import register
from .preprocess import *
from .postprocess import *


@register
class DetectionCustomOp(ModelBaseOp):
    def __init__(self, model_cfg, env_cfg):
        super(DetectionOp, self).__init__(model_cfg, env_cfg)
        self.model_cfg = model_cfg
        mod = importlib.import_module(__name__)
        self.preprocessor = create_operators(model_cfg["PreProcess"], mod)
        self.postprocessor = create_operators(model_cfg["PostProcess"], mod)

    @classmethod
    def get_output_keys(cls):
        return ["dt_bboxes", "dt_scores", "dt_class_ids", "dt_cls_names"]

    def preprocess(self, image):
        im_info = {
            'scale_factor': np.array(
                [1., 1.], dtype=np.float32),
            'im_shape': np.array(
                image.shape[:2], dtype=np.float32),
            'input_shape': self.model_cfg["image_shape"],
        }
        for ops in self.preprocessor:
            image, im_info = ops(image, im_info)
        return image, im_info

    def postprocess(self, inputs, result, bbox_num):
        outputs = result
        for idx, ops in enumerate(self.postprocessor):
            if idx == len(self.postprocessor) - 1:
                outputs, bbox_num = ops(outputs, bbox_num, self.output_keys)
            else:
                outputs, bbox_num = ops(outputs, bbox_num)
        return outputs, bbox_num

    def create_inputs(self, imgs, im_info):
        inputs = {}
        im_shape = []
        scale_factor = []
        if len(imgs) == 1:
            image = np.array((imgs[0], )).astype('float32')
            im_shape = np.array((im_info[0]['im_shape'], )).astype('float32')
            scale_factor = np.array(
                (im_info[0]['scale_factor'], )).astype('float32')
            inputs = dict(
                im_shape=im_shape, image=image, scale_factor=scale_factor)
            outputs = [inputs[key] for key in self.input_names]
            return outputs

        for e in im_info:
            im_shape.append(np.array((e['im_shape'], )).astype('float32'))
            scale_factor.append(
                np.array((e['scale_factor'], )).astype('float32'))

        inputs['im_shape'] = np.concatenate(im_shape, axis=0)
        inputs['scale_factor'] = np.concatenate(scale_factor, axis=0)

        imgs_shape = [[e.shape[1], e.shape[2]] for e in imgs]
        max_shape_h = max([e[0] for e in imgs_shape])
        max_shape_w = max([e[1] for e in imgs_shape])
        padding_imgs = []
        for img in imgs:
            im_c, im_h, im_w = img.shape[:]
            padding_im = np.zeros(
                (im_c, max_shape_h, max_shape_w), dtype=np.float32)
            padding_im[:, :im_h, :im_w] = img
            padding_imgs.append(padding_im)
        inputs['image'] = np.stack(padding_imgs, axis=0)
        outputs = [inputs[key] for key in self.input_names]
        return outputs

    def infer(self, image_list):
        inputs = []
        batch_loop_cnt = math.ceil(float(len(image_list)) / self.batch_size)
        results = []
        bbox_nums = []
        for i in range(batch_loop_cnt):
            start_index = i * self.batch_size
            end_index = min((i + 1) * self.batch_size, len(image_list))
            batch_image_list = image_list[start_index:end_index]
            # preprocess
            output_list = []
            info_list = []
            for img in batch_image_list:
                output, info = self.preprocess(img)
                output_list.append(output)
                info_list.append(info)
            inputs = self.create_inputs(output_list, info_list)

            # model inference
            result = self.predictor.run(inputs)
            res = result[0]
            bbox_num = result[1]
            # postprocess
            res, bbox_num = self.postprocess(inputs, res, bbox_num)
            results.append(res)
            bbox_nums.append(bbox_num)
        # results = self.merge_batch_result(results)
        return results, bbox_nums

    def __call__(self, inputs):
        """
        step1: parser inputs
        step2: run
        step3: merge results
        input: a list of dict
        """
        # for the input_keys as list
        # inputs = [pipe_input[key] for pipe_input in pipe_inputs for key in self.input_keys]

        key = self.input_keys[0]
        if isinstance(inputs[0][key], (list, tuple)):
            inputs = [input[key] for input in inputs]
        else:
            inputs = [[input[key]] for input in inputs]
        sub_index_list = [len(input) for input in inputs]
        inputs = reduce(lambda x, y: x.extend(y) or x, inputs)

        # step2: run
        outputs, bbox_nums = self.infer(inputs)

        # step3: merge
        curr_offsef_id = 0
        pipe_outputs = []
        for i, bbox_num in enumerate(bbox_nums):
            output = outputs[i]
            start_id = 0
            for num in bbox_num:
                end_id = start_id + num
                out = {k: v[start_id:end_id] for k, v in output.items()}
                pipe_outputs.append(out)
                start_id = end_id
        return pipe_outputs
