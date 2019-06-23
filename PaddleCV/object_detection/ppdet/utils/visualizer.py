# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import logging
import numpy as np
import pycocotools.mask as mask_util
from PIL import Image, ImageDraw

from .colormap import colormap

logger = logging.getLogger(__name__)

__all__ = ['visualize_results']

SAVE_HOME = 'output'


def visualize_results(image_path,
                      catid2name,
                      threshold=0.5,
                      bbox_results=None,
                      mask_results=None):
    """
    TODO(dengkaipeng): add more comments
    Visualize bbox and mask results
    """
    image = None
    if mask_results:
        image = draw_mask(image_path, mask_results, threshold)
    if bbox_results:
        draw_bbox(image_path, catid2name, bbox_results, threshold, image)


def draw_mask(image_path, segms, threshold, alpha=0.7, save_image=False):
    """
    TODO(dengkaipeng): add more comments
    Draw mask on image
    """
    image = Image.open(image_path)
    im_width, im_height = image.size
    mask_color_id = 0
    w_ratio = .4
    image = np.array(image).astype('float32')
    for dt in np.array(segms):
        segm, score = dt['segmentation'], dt['score']
        if score < threshold:
            continue
        mask = mask_util.decode(segm) * 255
        color_list = colormap(rgb=True)
        color_mask = color_list[mask_color_id % len(color_list), 0:3]
        mask_color_id += 1
        for c in range(3):
            color_mask[c] = color_mask[c] * (1 - w_ratio) + w_ratio * 255
        idx = np.nonzero(mask)
        image[idx[0], idx[1], :] *= 1.0 - alpha
        image[idx[0], idx[1], :] += alpha * color_mask
    image = Image.fromarray(image.astype('uint8'))

    if not os.path.exists(SAVE_HOME):
        os.makedirs(SAVE_HOME)
    if save_image:
        save_name = get_save_image_name(image_path)
        logger.info("Detection mask results save in {}".format(save_name))
        image.save(save_name)
    return image


def draw_bbox(image_path, catid2name, bboxes, threshold, image=None):
    """
    TODO(dengkaipeng): add more comments
    Draw bbox on image
    """
    if image is None:
        image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size

    for dt in np.array(bboxes):
        catid, bbox, score = dt['category_id'], dt['bbox'], dt['score']
        if score < threshold:
            continue
        xmin, ymin, w, h = bbox
        xmax = xmin + w
        ymax = ymin + h
        draw.line(
            [(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin),
             (xmin, ymin)],
            width=2,
            fill='red')
        if image.mode == 'RGB':
            draw.text((xmin, ymin), catid2name[catid], (255, 255, 0))

    if not os.path.exists(SAVE_HOME):
        os.makedirs(SAVE_HOME)
    save_name = get_save_image_name(image_path)
    logger.info("Detection bbox results save in {}".format(save_name))
    image.save(save_name)


def get_save_image_name(image_path):
    """
    Get save image name from source image path.
    """
    image_name = image_path.split('/')[-1]
    name, ext = os.path.splitext(image_name)
    return os.path.join(SAVE_HOME, "{}".format(name)) + ext
