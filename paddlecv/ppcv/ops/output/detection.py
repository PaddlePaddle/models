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
from PIL import Image, ImageDraw, ImageFile

logger = setup_logger('DetOutput')


def get_id_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
    return color


def get_color_map_list(num_classes):
    """
    Args:
        num_classes (int): number of class
    Returns:
        color_map (list): RGB color list
    """
    color_map = num_classes * [0, 0, 0]
    for i in range(0, num_classes):
        j = 0
        lab = i
        while lab:
            color_map[i * 3] |= (((lab >> 0) & 1) << (7 - j))
            color_map[i * 3 + 1] |= (((lab >> 1) & 1) << (7 - j))
            color_map[i * 3 + 2] |= (((lab >> 2) & 1) << (7 - j))
            j += 1
            lab >>= 3
    color_map = [color_map[i:i + 3] for i in range(0, len(color_map), 3)]
    return color_map


def draw_det(image, dt_bboxes, dt_scores, dt_cls_names, input_id=None):
    im = Image.fromarray(image[:, :, ::-1])
    draw_thickness = min(im.size) // 320
    draw = ImageDraw.Draw(im)
    name_set = sorted(set(dt_cls_names))
    name2clsid = {name: i for i, name in enumerate(name_set)}
    clsid2color = {}
    color_list = get_color_map_list(len(name_set))

    for i in range(len(dt_bboxes)):
        box, score, name = dt_bboxes[i], dt_scores[i], dt_cls_names[i]
        if input_id is None:
            color = tuple(color_list[name2clsid[name]])
        else:
            color = get_id_color(input_id[i])

        xmin, ymin, xmax, ymax = box
        # draw bbox
        draw.line(
            [(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin),
             (xmin, ymin)],
            width=draw_thickness,
            fill=color)

        # draw label
        text = "{} {:.4f}".format(name, score)
        box = draw.textbbox((xmin, ymin), text, anchor='lt')
        draw.rectangle(box, fill=color)
        draw.text((box[0], box[1]), text, fill=(255, 255, 255))
    image = np.array(im)
    return image


@register
class DetOutput(OutputBaseOp):
    def __init__(self, model_cfg, env_cfg):
        super(DetOutput, self).__init__(model_cfg, env_cfg)

    def __call__(self, inputs):
        total_res = []
        for res in inputs:
            fn, image, dt_bboxes, dt_scores, dt_cls_names = res.values()
            image = draw_det(image, dt_bboxes, dt_scores, dt_cls_names)
            res.pop('input.image')
            if self.frame_id != -1:
                res.update({'frame_id': frame_id})
            logger.info(res)
            if self.save_img:
                file_name = os.path.split(fn)[-1]
                out_path = os.path.join(self.output_dir, file_name)
                logger.info('Save output image to {}'.format(out_path))
                cv2.imwrite(out_path, image)
            if self.save_res or self.return_res:
                total_res.append(res)
        if self.save_res:
            res_file_name = 'det_output.json'
            out_path = os.path.join(self.output_dir, res_file_name)
            logger.info('Save output result to {}'.format(out_path))
            with open(out_path, 'w') as f:
                json.dump(total_res, f)
        if self.return_res:
            return total_res
        return
