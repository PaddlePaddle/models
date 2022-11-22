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
import math
import glob
import json
from collections import defaultdict

import cv2
import paddle
import numpy as np
from PIL import Image

from ppcv.utils.logger import setup_logger
from ppcv.core.workspace import register

from .base import OutputBaseOp

logger = setup_logger('SegOutput')


@register
class SegOutput(OutputBaseOp):
    def __init__(self, model_cfg, env_cfg):
        super().__init__(model_cfg, env_cfg)

    def __call__(self, inputs):
        total_res = []
        for input in inputs:
            fn, _, seg_map = input.values()
            res = dict(filename=fn, seg_map=seg_map.tolist())
            if self.save_res or self.return_res:
                total_res.append(res)

            if self.save_img:
                seg_map = get_pseudo_color_map(seg_map)
                file_name = os.path.split(fn)[-1]
                out_path = os.path.join(self.output_dir, file_name)
                seg_map.save(out_path)
                logger.info('Save output image to {}'.format(out_path))

        if self.save_res:
            res_file_name = 'seg_output.json'
            out_path = os.path.join(self.output_dir, res_file_name)
            with open(out_path, 'w') as f:
                json.dump(total_res, f)
            logger.info('Save output result to {}'.format(out_path))

        if self.return_res:
            return total_res


@register
class HumanSegOutput(OutputBaseOp):
    def __init__(self, model_cfg, env_cfg):
        super().__init__(model_cfg, env_cfg)

    def __call__(self, inputs):
        total_res = []
        for input in inputs:
            fn, img, seg_map = input.values()
            res = dict(filename=fn, seg_map=seg_map.tolist())
            if self.save_res or self.return_res:
                total_res.append(res)

            if self.save_img:
                alpha = seg_map[1]
                alpha = cv2.resize(alpha, (img.shape[1], img.shape[0]))
                alpha = (alpha * 255).astype('uint8')
                img = img[:, :, ::-1]
                res_img = np.concatenate(
                    [img, alpha[:, :, np.newaxis]], axis=-1)

                filename = os.path.basename(fn).split('.')[0]
                out_path = os.path.join(self.output_dir, filename + ".png")
                cv2.imwrite(out_path, res_img)
                logger.info('Save output image to {}'.format(out_path))

        if self.save_res:
            res_file_name = 'humanseg_output.json'
            out_path = os.path.join(self.output_dir, res_file_name)
            with open(out_path, 'w') as f:
                json.dump(total_res, f)
            logger.info('Save output result to {}'.format(out_path))

        if self.return_res:
            return total_res


@register
class MattingOutput(OutputBaseOp):
    def __init__(self, model_cfg, env_cfg):
        super().__init__(model_cfg, env_cfg)

    def __call__(self, inputs):
        total_res = []
        for input in inputs:
            fn, img, seg_map = input.values()
            res = dict(filename=fn, seg_map=seg_map.tolist())
            if self.save_res or self.return_res:
                total_res.append(res)

            if self.save_img:
                alpha = seg_map.squeeze()
                alpha = cv2.resize(alpha, (img.shape[1], img.shape[0]))
                alpha = (alpha * 255).astype('uint8')

                filename = os.path.basename(fn).split('.')[0]
                out_path = os.path.join(self.output_dir, filename + ".png")
                cv2.imwrite(out_path, alpha)
                logger.info('Save output image to {}'.format(out_path))

        if self.save_res:
            res_file_name = 'matting_output.json'
            out_path = os.path.join(self.output_dir, res_file_name)
            with open(out_path, 'w') as f:
                json.dump(total_res, f)
            logger.info('Save output result to {}'.format(out_path))

        if self.return_res:
            return total_res


def get_pseudo_color_map(pred, color_map=None):
    """
    Get the pseudo color image.

    Args:
        pred (numpy.ndarray): the origin predicted image.
        color_map (list, optional): the palette color map. Default: None,
            use paddleseg's default color map.

    Returns:
        (numpy.ndarray): the pseduo image.
    """
    pred_mask = Image.fromarray(pred.astype(np.uint8), mode='P')
    if color_map is None:
        color_map = get_color_map_list(256)
    pred_mask.putpalette(color_map)
    return pred_mask


def get_color_map_list(num_classes, custom_color=None):
    """
    Returns the color map for visualizing the segmentation mask,
    which can support arbitrary number of classes.

    Args:
        num_classes (int): Number of classes.
        custom_color (list, optional): Save images with a custom color map. Default: None, use paddleseg's default color map.

    Returns:
        (list). The color map.
    """

    num_classes += 1
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
    color_map = color_map[3:]

    if custom_color:
        color_map[:len(custom_color)] = custom_color
    return color_map
