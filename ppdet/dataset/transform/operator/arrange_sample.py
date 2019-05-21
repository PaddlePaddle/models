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

# function:
#    operators to process sample,
#    eg: decode/resize/crop image

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import uuid
import logging
import random
import math
import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageDraw
from functools import reduce
from .base import BaseOperator, registered_ops, register_op

logger = logging.getLogger(__name__)

@register_op
class ArrangeRCNN(BaseOperator):
    """Transform the sample dict to the sample tuple
       which the model need when training.
    """
    def __init__(self, is_mask=False):
        """ Get the standard output.
        Args:
            is_mask (bool): confirm whether to use mask rcnn
        """
        super(ArrangeRCNN, self).__init__()
        self.is_mask = is_mask
        if not (isinstance(self.is_mask, bool)):
            raise TypeError('{}: the input type is error.'
                            .format(self.__str__))

    def __call__(self, sample, context=None):
        """
        Input:
            sample: a dict which contains image
                    info and annotation info.
            context: a dict which contains additional info.
        Output:
            sample: a tuple which contains the
                    info which training model need.
                    tupe is (image, gt_bbox, gt_class, is_crowd, im_info, gt_masks)
        """
        im = sample['image']
        gt_bbox = sample['gt_bbox']
        gt_class = sample['gt_class']
        outs = (im, gt_bbox, gt_class)
        keys = list(sample.keys())
        if 'is_crowd' in keys:
            is_crowd = sample['is_crowd']
            outs = outs + (is_crowd,)
        if 'im_info' in keys:
            im_info = sample['im_info']
            outs = outs + (im_info,)
        im_id = sample['im_id']
        outs = outs + (im_id,)
        gt_masks = []
        if self.is_mask and len(sample['gt_poly']) != 0 \
                and 'is_crowd' in keys:
            valid = True
            segms = sample['gt_poly']
            assert len(segms) == is_crowd.shape[0]
            for i in range(len(sample['gt_poly'])):
                segm, iscrowd = segms[i], is_crowd[i]
                gt_segm = []
                if iscrowd:
                    gt_segm.append([[0, 0]])
                else:
                    for poly in segm:
                        if len(poly) == 0:
                            valid = False
                            break
                        gt_segm.append(np.array(poly).reshape(-1, 2))
                if (not valid) or len(gt_segm) == 0:
                    break
                gt_masks.append(gt_segm)
            outs = outs + (gt_masks, )
        return outs

@register_op   
class ArrangeSSD(BaseOperator):
    """Transform the sample dict to the sample tuple
       which the model need when training.
    """
    def __init__(self, is_mask=False):
        """ Get the standard output.
        Args:
            is_mask (bool): confirm whether to use mask rcnn
        """
        super(ArrangeSSD, self).__init__()
        self.is_mask = is_mask
        if not (isinstance(self.is_mask, bool)):
            raise TypeError('{}: the input type is error.'
                            .format(self.__str__))

    def __call__(self, sample, context=None):
        """
        Input:
            sample: a dict which contains image
                    info and annotation info.
            context: a dict which contains additional info.
        Output:
            sample: a tuple which contains the
                    info which training model need.
                    tupe is (image, gt_bbox, gt_class, is_crowd, im_info, gt_masks)
        """
        im = sample['image']
        gt_bbox = sample['gt_bbox']
        gt_class = sample['gt_class']
        difficult = sample['difficult']
        outs = (im, gt_bbox, gt_class, difficult)
        return outs
