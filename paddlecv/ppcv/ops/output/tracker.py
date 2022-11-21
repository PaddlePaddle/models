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
from .detection import draw_det
from ppcv.utils.logger import setup_logger
from ppcv.core.workspace import register
from PIL import Image, ImageDraw, ImageFile

logger = setup_logger('TrackerOutput')


def write_mot_results(filename, results, data_type='mot', num_classes=1):
    # support single and multi classes
    if data_type in ['mot', 'mcmot']:
        save_format = '{frame},{id},{x1},{y1},{w},{h},{score},{cls_id},-1,-1\n'
    elif data_type == 'kitti':
        save_format = '{frame} {id} car 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'
    else:
        raise ValueError(data_type)

    frame_id, tk_bboxes, tk_scores, tk_ids, tk_cls_ids = results
    frame_id = -1 if data_type == 'kitti' else frame_id
    with open(filename, 'w') as f:
        for bbox, score, tk_id, cls_id in zip(tk_bboxes, tk_scores, tk_ids,
                                              tk_cls_ids):
            if tk_id < 0: continue
            if data_type == 'mot':
                cls_id = -1

            x1, y1, x2, y2 = bbox
            w, h = x2 - x1, y2 - y1
            line = save_format.format(
                frame=frame_id,
                id=tk_id,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                w=w,
                h=h,
                score=score,
                cls_id=cls_id)
            f.write(line)


@register
class TrackerOutput(OutputBaseOp):
    def __init__(self, model_cfg, env_cfg):
        super(TrackerOutput, self).__init__(model_cfg, env_cfg)

    def __call__(self, inputs):
        total_res = []
        vis_images = []
        for res in inputs:
            fn, image, tk_bboxes, tk_scores, tk_ids, tk_cls_ids, tk_cls_names = res.values(
            )
            tk_names = [
                '{} {}'.format(tk_cls_name, tk_id)
                for tk_id, tk_cls_name in zip(tk_ids, tk_cls_names)
            ]
            image = draw_det(image, tk_bboxes, tk_scores, tk_names, tk_ids)
            res.pop('input.image')
            if self.frame_id != -1:
                res.update({'frame_id': self.frame_id})
            logger.info(res)
            if self.save_img:
                vis_images.append(image)
            if self.save_res or self.return_res:
                total_res.append(res)
        if self.save_res:
            video_name = fn.split('/')[-1].split('.')[0]
            output_dir = os.path.join(self.output_dir, video_name)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            out_path = os.path.join(output_dir, '{}.txt'.format(self.frame_id))
            logger.info('Save output result to {}'.format(out_path))
            write_mot_results(
                out_path,
                [self.frame_id, tk_bboxes, tk_scores, tk_ids, tk_cls_ids])
        if self.return_res:
            if vis_images:
                for i, vis_im in enumerate(vis_images):
                    total_res[i].update({'output': vis_im})
            return total_res
        return
