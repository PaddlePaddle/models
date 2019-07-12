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

from ..data.source.voc_loader import pascalvoc_label
from .coco_eval import bbox2out

import logging
logger = logging.getLogger(__name__)

__all__ = [
    'bbox_eval', 'bbox2out', 'get_category_info'
]


def bbox_eval(results, 
              class_num, 
              overlap_thesh=0.5, 
              map_type='11point',
              is_bbox_normalized=False,
              evaluate_difficult=False):
    assert 'bbox' in results[0]
    logger.info("Start evaluate...")

    class_scores = [[] for i in range(class_num)]
    class_gt_cnts = [0] * class_num
    for t in results:
        bboxes = t['bbox'][0]
        bbox_lengths = t['bbox'][1][0]

        if bboxes.shape == (1, 1) or bboxes is None:
            continue

        gt_boxes = t['gt_box'][0]
        gt_box_lengths = t['gt_box'][1][0]
        gt_labels = t['gt_label'][0]
        difficults = t['is_difficult'][0]
        assert len(gt_boxes) == len(gt_labels) == len(difficults)

        for i, gt_label in enumerate(gt_labels):
            if evaluate_difficult or int(difficults[i]) == 0:
                class_gt_cnts[int(gt_label[0])] += 1

        bbox_idx = 0
        gt_box_idx = 0
        for i in range(len(bbox_lengths)):
            bbox_num = bbox_lengths[i]
            gt_box_num = gt_box_lengths[i]
            bbox = bboxes[bbox_idx: bbox_idx + bbox_num]
            gt_box = gt_boxes[gt_box_idx: gt_box_idx + gt_box_num]
            gt_label = gt_labels[gt_box_idx: gt_box_idx + gt_box_num]
            difficult = difficults[gt_box_idx: gt_box_idx + gt_box_num]
            cls_scores = single_map(bbox, gt_box, gt_label, difficult,
                                    class_num, overlap_thesh,
                                    is_bbox_normalized, 
                                    evaluate_difficult)
            for c in range(class_num):
                class_scores[c].extend(cls_scores[c])
            bbox_idx += bbox_num
            gt_box_idx += gt_box_num

    def get_tp_fp_accum(score_list):
        sorted_list = sorted(score_list, key=lambda s: s[0], reverse=True)
        accum_tp = 0
        accum_fp = 0
        accum_tp_list = []
        accum_fp_list = []
        for (score, pos) in sorted_list:
            accum_tp += int(pos)
            accum_tp_list.append(accum_tp)
            accum_fp += 1 - int(pos)
            accum_fp_list.append(accum_fp)
        return accum_tp_list, accum_fp_list

    logger.info("Accumulate evaluatation results...")
    mAP = 0.
    valid_cnt = 0
    for score, cnt in zip(class_scores, class_gt_cnts):
        if cnt == 0 or len(score) == 0:
            continue

        accum_tp_list, accum_fp_list = get_tp_fp_accum(score)
        precision = []
        recall = []
        for ac_tp, ac_fp in zip(accum_tp_list, accum_fp_list):
            precision.append(float(ac_tp) / (ac_tp + ac_fp))
            recall.append(float(ac_tp) / cnt)

        if map_type == '11point':
            max_precisions = [0.] * 11
            start_idx = len(precision) - 1
            for j in range(10, -1, -1):
                for i in range(start_idx, -1, -1):
                    if recall[i] < float(j) / 10.:
                        start_idx = i
                        if j > 0:
                            max_precisions[j - 1] = max_precisions[j]
                            break
                    else:
                        if max_precisions[j] < precision[i]:
                            max_precisions[j] = precision[i]
            mAP += sum(max_precisions) / 11. 
            valid_cnt += 1
        elif map_type == 'integral':
            import math
            ap = 0.
            prev_recall = 0.
            for i in range(len(precision)):
                recall_gap = math.fabs(recall[i] - prev_recall)
                if recall_gap > 1e-6:
                    ap += precision[i] * recall_gap
                    prev_recall = recall[i]
            mAP += ap
            valid_cnt += 1
        else:
            logger.error("Unspported mAP type {}".format(map_type))
            sys.exit(1)

    if valid_cnt > 0:
        mAP /= float(valid_cnt)

    logger.info("mAP({:.2f}, {}) = {:.3f}".format(overlap_thesh, map_type, mAP))


def single_map(box, 
               gt_box, 
               gt_label, 
               difficult,
               class_num, 
               overlap_thesh, 
               is_bbox_normalized,
               evaluate_difficult):
    cls_scores = [[] for i in range(class_num)]
    visited = [False] * len(gt_label)
    for b in box:
        label, score, xmin, ymin, xmax, ymax = b.tolist()
        pred = [xmin, ymin, xmax, ymax]
        max_idx = -1
        max_overlap = -1.0
        for i, gl in enumerate(gt_label):
            if int(gl) == int(label):
                overlap = jaccard_overlap(pred, gt_box[i], is_bbox_normalized)
                if overlap > max_overlap:
                    max_overlap = overlap
                    max_idx = i

        if max_overlap > overlap_thesh:
            if evaluate_difficult or int(difficult[max_idx]) == 0:
                if not visited[max_idx]:
                    cls_scores[int(label)].append([score, 1.0])
                    visited[max_idx] = True
                else:
                    cls_scores[int(label)].append([score, 0.0])
        else:
            cls_scores[int(label)].append([score, 0.0])

    return cls_scores


def jaccard_overlap(pred, gt, is_bbox_normalized=False):
    if pred[0] >= gt[2] or pred[2] <= gt[0] or \
        pred[1] >= gt[3] or pred[3] <= gt[1]:
        return 0.
    inter_xmin = max(pred[0], gt[0])
    inter_ymin = max(pred[1], gt[1])
    inter_xmax = min(pred[2], gt[2])
    inter_ymax = min(pred[3], gt[3])
    inter_size = bbox_area([inter_xmin, inter_ymin,
                            inter_xmax, inter_ymax],
                            is_bbox_normalized)
    pred_size = bbox_area(pred, is_bbox_normalized)
    gt_size = bbox_area(gt, is_bbox_normalized)
    overlap = float(inter_size) / (
        pred_size + gt_size - inter_size)
    return overlap


def bbox_area(bbox, is_bbox_normalized):
    norm = 1. - float(is_bbox_normalized)
    width = bbox[2] - bbox[0] + norm
    height = bbox[3] - bbox[1] + norm
    return width * height


def get_category_info(anno_file=None,
                      with_background=True,
                      use_default_label=False):
    if use_default_label or anno_file is None \
            or not os.path.exists(anno_file):
        logger.info("Not found annotation file {}, load "
                    "voc2012 categories.".format(anno_file))
        return vocall_category_info(with_background)
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


def vocall_category_info(with_background=True):
    """
    Get class id to category id map and category id
    to category name map of mixup voc dataset

    Args:
        with_background (bool, default True):
            whether load background as class 0.
    """
    label_map = pascalvoc_label(with_background)
    label_map = sorted(label_map.items(), key=lambda x: x[1])
    cats = [l[0] for l in label_map]

    if with_background:
        cats.insert(0, 'background')

    clsid2catid = {i: i for i in range(len(cats))}
    catid2name = {i: name for i, name in enumerate(cats)}

    return clsid2catid, catid2name
