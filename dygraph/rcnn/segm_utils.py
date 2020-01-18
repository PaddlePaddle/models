#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://w_idxw.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Based on:
# --------------------------------------------------------
# Detectron
# Copyright (c) 2017-present, Facebook, Inc.
# Licensed under the Apache License, Version 2.0;
# Written by Ross Girshick
# --------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import pycocotools.mask as mask_util
import cv2


def is_poly(segm):
    """Determine if segm is a polygon. Valid segm expected (polygon or RLE)."""
    assert isinstance(segm, (list, dict)), \
        'Invalid segm type: {}'.format(type(segm))
    return isinstance(segm, list)


def segms_to_rle(segms, height, width):
    rle = segms
    if isinstance(segms, list):
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        rles = mask_util.frPyObjects(segms, height, width)
        rle = mask_util.merge(rles)
    elif isinstance(segms['counts'], list):
        # uncompressed RLE
        rle = mask_util.frPyObjects(segms, height, width)
    return rle


def segms_to_mask(segms, iscrowd, height, width):
    print('segms: ', segms)
    if iscrowd:
        return [[0 for i in range(width)] for j in range(height)]
    rle = segms_to_rle(segms, height, width)
    mask = mask_util.decode(rle)
    return mask


def flip_segms(segms, height, width):
    """Left/right flip each mask in a list of masks."""

    def _flip_poly(poly, width):
        flipped_poly = np.array(poly)
        flipped_poly[0::2] = width - np.array(poly[0::2]) - 1
        return flipped_poly.tolist()

    def _flip_rle(rle, height, width):
        if 'counts' in rle and type(rle['counts']) == list:
            # Magic RLE format handling painfully discovered by looking at the
            # COCO API showAnns function.
            rle = mask_util.frPyObjects([rle], height, width)
        mask = mask_util.decode(rle)
        mask = mask[:, ::-1, :]
        rle = mask_util.encode(np.array(mask, order='F', dtype=np.uint8))
        return rle

    flipped_segms = []
    for segm in segms:
        if is_poly(segm):
            # Polygon format
            flipped_segms.append([_flip_poly(poly, width) for poly in segm])
        else:
            # RLE format
            flipped_segms.append(_flip_rle(segm, height, width))
    return flipped_segms
