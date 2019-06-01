#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import json
import cv2
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import pycocotools.mask as mask_util

import logging
logger = logging.getLogger(__name__)

__all__ = ['bbox_eval', 'mask_eval']


def coco_eval(results, anno_file, outfile, with_background=True):
    coco_gt = COCO(anno_file)
    cat_ids = coco_gt.getCatIds()

    # when with_background = True, mapping category to classid, like:
    #   background:0, first_class:1, second_class:2, ...
    clsid2catid = dict(
            {i + int(with_background): catid
                for i, catid in enumerate(cat_ids)})

    if 'bbox' in results[0]:
        xywh_results = bbox2out(results, clsid2catid)

        assert outfile.endswith('.json')
        with open(outfile, 'w') as f:
            json.dump(xywh_results, f)

        logger.info("Start evaluate...")
        coco_dt = coco_gt.loadRes(outfile)
        coco_ev = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_ev.evaluate()
        coco_ev.accumulate()
        coco_ev.summarize()


def mask_eval(results, anno_file, outfile, resolution, thresh_binarize=0.5):
    assert 'mask' in results[0]

    coco_gt = COCO(anno_file)
    clsid2catid = {i + 1: v for i, v in enumerate(coco_gt.getCatIds())}

    segm_results = mask2out(results, clsid2catid, resolution, thresh_binarize)
    with open("segms.json", 'w') as f:
        json.dump(segm_results, f)

    logger.info("Start evaluate...")
    coco_dt = coco_gt.loadRes("segms.json")
    coco_ev = COCOeval(coco_gt, coco_dt, 'segm')
    coco_ev.evaluate()
    coco_ev.accumulate()
    coco_ev.summarize()


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
                catid = clsid2catid[clsid]
                w = xmax - xmin + 1
                h = ymax - ymin + 1
                bbox = [xmin, ymin, w, h]
                coco_res = {
                    'image_id': im_id,
                    'category_id': catid,
                    'bbox': bbox,
                    'score': score
                }
                xywh_res.append(coco_res)
                k += 1
    return xywh_res


def mask2out(results, clsid2catid, resolution, thresh_binarize=0.5):
    scale = (resolution + 2.0) / resolution

    segm_res = []

    # for each batch
    for t in results:
        bboxes = t['bbox'][0]

        lengths = t['bbox'][1][0]
        im_ids = np.array(t['im_id'][0])
        if bboxes.shape == (1, 1) or bboxes is None:
            continue
        if len(bboxes.tolist()) == 0:
            continue

        masks = t['mask'][0]
        im_info = t['im_info'][0][0]

        k = 0
        s = 0
        segms_results = [[] for _ in range(len(lengths))]
        # for each sample
        for i in range(len(lengths)):
            num = lengths[i]
            im_id = int(im_ids[i][0])

            bbox = bboxes[s:s + num][:, 2:]
            clsid_scores = bboxes[s:s + num][:, 0:2]
            mask = masks[s:s + num]
            s += num

            im_h = int(round(im_info[0] / im_info[2]))
            im_w = int(round(im_info[1] / im_info[2]))

            expand_bbox = expand_boxes(bbox, scale)
            expand_bbox = expand_bbox.astype(np.int32)

            padded_mask = np.zeros(
                (resolution + 2, resolution + 2), dtype=np.float32)

            cls_segms = []
            for j in range(num):
                xmin, ymin, xmax, ymax = expand_bbox[j].tolist()
                clsid, score = clsid_scores[j].tolist()
                clsid = int(clsid)
                padded_mask[1:-1, 1:-1] = mask[j, clsid, :, :]

                catid = clsid2catid[clsid]

                w = xmax - xmin + 1
                h = ymax - ymin + 1
                w = np.maximum(w, 1)
                h = np.maximum(h, 1)

                resized_mask = cv2.resize(padded_mask, (w, h))
                resized_mask = np.array(
                    resized_mask > thresh_binarize, dtype=np.uint8)
                im_mask = np.zeros((im_h, im_w), dtype=np.uint8)

                x0 = max(xmin, 0)
                x1 = min(xmax + 1, im_w)
                y0 = max(ymin, 0)
                y1 = min(ymax + 1, im_h)

                im_mask[y0:y1, x0:x1] = resized_mask[(y0 - ymin):(y1 - ymin), (
                    x0 - xmin):(x1 - xmin)]
                segm = mask_util.encode(
                    np.array(
                        im_mask[:, :, np.newaxis], order='F'))[0]
                catid = clsid2catid[clsid]
                segm['counts'] = segm['counts'].decode('utf8')
                coco_res = {
                    'image_id': im_id,
                    'category_id': catid,
                    'segmentation': segm,
                    'score': score
                }
                segm_res.append(coco_res)
    return segm_res


def expand_boxes(boxes, scale):
    """
    Expand an array of boxes by a given scale.
    """
    w_half = (boxes[:, 2] - boxes[:, 0]) * .5
    h_half = (boxes[:, 3] - boxes[:, 1]) * .5
    x_c = (boxes[:, 2] + boxes[:, 0]) * .5
    y_c = (boxes[:, 3] + boxes[:, 1]) * .5

    w_half *= scale
    h_half *= scale

    boxes_exp = np.zeros(boxes.shape)
    boxes_exp[:, 0] = x_c - w_half
    boxes_exp[:, 2] = x_c + w_half
    boxes_exp[:, 1] = y_c - h_half
    boxes_exp[:, 3] = y_c + h_half

    return boxes_exp
