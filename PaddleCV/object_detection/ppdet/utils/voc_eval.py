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
import sys
import numpy as np

import logging
logger = logging.getLogger(__name__)

__all__ = [
    'bbox2out', 'get_category_info'
]


def clip_bbox(bbox):
    xmin = max(min(bbox[0], 1.), 0.)
    ymin = max(min(bbox[1], 1.), 0.)
    xmax = max(min(bbox[2], 1.), 0.)
    ymax = max(min(bbox[3], 1.), 0.)
    return xmin, ymin, xmax, ymax


def bbox2out(results, clsid2catid):
    xywh_res = []
    for t in results:
        bboxes = t['bbox'][0]
        lengths = t['bbox'][1][0]
        im_ids = np.array(t['im_id'][0])
        if bboxes.shape == (1, 1) or bboxes is None:
            continue

        k = 0
        for i in range(len(lengths)):
            num = lengths[i]
            im_id = int(im_ids[i][0])
            for j in range(num):
                dt = bboxes[k]
                clsid, score, xmin, ymin, xmax, ymax = dt.tolist()
                xmin, ymin, xmax, ymax = \
                        clip_bbox([xmin, ymin, xmax, ymax])
                catid = clsid2catid[clsid]
                w = xmax - xmin
                h = ymax - ymin
                bbox = [xmin, ymin, w, h]
                res = {
                    'image_id': im_id,
                    'category_id': catid,
                    'bbox': bbox,
                    'score': score
                }
                xywh_res.append(res)
                k += 1
    return xywh_res


def get_category_info(anno_file=None, with_background=True):
    if anno_file is None or not os.path.exists(anno_file):
        logger.info("Not found annotation file {}, load "
                    "voc2012 categories.".format(anno_file))
        return voc2012_category_info(with_background)
    else:
        logger.info("Load categories from {}".format(anno_file))
        return get_category_info_from_anno(anno_file, with_background)


def get_category_info_from_anno(anno_file, with_background=True):
    """
    Get class id to category id map and category id
    to category name map from annotation file.

    Args:
        anno_file (str): annotation file path
        with_background (bool, default True):
            whether load background as class 0.
    """
    cats = []
    with open(anno_file) as f:
        for line in f.readlines():
            cats.append(line.strip())

    if cats[0] != 'background' and with_background:
        cats.insert(0, 'background')
    if cats[0] == 'background' and not with_background:
        cats = cats[1:]

    clsid2catid = {i: i for i in range(len(cats))}
    catid2name = {i: name for i, name in enumerate(cats)}

    return clsid2catid, catid2name


def voc2012_category_info(with_background=True):
    """
    Get class id to category id map and category id
    to category name map of voc2017 dataset

    Args:
        with_background (bool, default True):
            whether load background as class 0.
    """

    cats = ['sheep'
            'horse'
            'bicycle'
            'aeroplane'
            'cow'
            'car'
            'dog'
            'bus'
            'cat'
            'person'
            'train'
            'diningtable'
            'bottle'
            'sofa'
            'pottedplant'
            'tvmonitor'
            'chair'
            'bird'
            'boat'
            'motorbike']

    if with_background:
        cats.insert(0, 'background')

    clsid2catid = {i: i for i in range(len(cats))}
    catid2name = {i: name for i, name in enumerate(cats)}

    return clsid2catid, catid2name
