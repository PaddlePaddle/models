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

import os
import numpy as np

from ppdet.data.source.widerface_loader import widerface_label
from ppdet.utils.coco_eval import bbox2out

import logging
logger = logging.getLogger(__name__)

__all__ = ['widerface_eval', 'bbox2out', 'get_category_info']


def cal_iou(rect1, rect2):
    lt_x = max(rect1[0], rect2[0])
    lt_y = max(rect1[1], rect2[1])
    rb_x = min(rect1[2], rect2[2])
    rb_y = min(rect1[3], rect2[3])
    if (rb_x > lt_x) and (rb_y > lt_y):
        intersection = (rb_x - lt_x) * (rb_y - lt_y)
    else:
        return 0

    area1 = (rect1[2] - rect1[0]) * (rect1[3] - rect1[1])
    area2 = (rect2[2] - rect2[0]) * (rect2[3] - rect2[1])

    intersection = min(intersection, area1, area2)
    union = area1 + area2 - intersection
    return float(intersection) / union


def widerface_eval(
        eval_results,
        output_eval_dir,
        output_fname='pred_result.txt', ):
    """
    Calculate ap according to prediction result list `eval_results`
    """

    def is_same_face(face_gt, face_pred):
        iou = cal_iou(face_gt, face_pred)
        return iou >= 0.3

    def eval_single_image(faces_gt, faces_pred):
        pred_is_true = [False] * len(faces_pred)
        gt_been_pred = [False] * len(faces_gt)
        for i in range(len(faces_pred)):
            isFace = False
            for j in range(len(faces_gt)):
                if gt_been_pred[j] == 0:
                    isFace = is_same_face(faces_gt[j], faces_pred[i][2:])
                    if isFace == 1:
                        gt_been_pred[j] = True
                        break
            pred_is_true[i] = isFace
        return pred_is_true

    faces_num_gt = 0
    score_res_pair = {}
    for t in eval_results:
        bboxes = t['bbox'][0]
        if bboxes.shape == (1, 1) or bboxes is None:
            continue
        bboxes = bboxes.tolist()

        gt_boxes = t['gt_box'][0].tolist()
        gt_box_lengths = t['gt_box'][1][0]
        faces_num_gt += np.sum(gt_box_lengths)

        pred_is_true = eval_single_image(gt_boxes, bboxes)
        for i in range(0, len(pred_is_true)):
            nowScore = bboxes[i][1]
            if score_res_pair.has_key(nowScore):
                score_res_pair[nowScore].append(int(pred_is_true[i]))
            else:
                score_res_pair[nowScore] = [int(pred_is_true[i])]
    keys = score_res_pair.keys()
    keys.sort(reverse=True)
    res_file = output_fname
    if output_eval_dir != None:
        res_file = os.path.join(output_eval_dir, output_fname)
    outfile = open(res_file, 'w')
    tp_num = 0
    predict_num = 0
    precision_list = []
    recall_list = []
    outfile.write("recall falsePositiveNum precision scoreThreshold\n")
    for i in range(len(keys)):
        k = keys[i]
        v = score_res_pair[k]
        predict_num += len(v)
        tp_num += sum(v)
        fp_num = predict_num - tp_num
        recall = float(tp_num) / faces_num_gt
        precision = float(tp_num) / predict_num
        outfile.write('{} {} {} {}\n'.format(recall, fp_num, precision, k))
        precision_list.append(float(tp_num) / predict_num)
        recall_list.append(recall)
    ap = precision_list[0] * recall_list[0]
    for i in range(1, len(precision_list)):
        ap += precision_list[i] * (recall_list[i] - recall_list[i - 1])
    outfile.write('AP={}\n'.format(ap))

    logger.info(
        "AP = {}\nFor more details, please checkout the evaluation res at {}"
        .format(ap, res_file))
    outfile.close()
    return ap


def get_category_info(anno_file=None,
                      with_background=True,
                      use_default_label=False):
    if use_default_label or anno_file is None \
            or not os.path.exists(anno_file):
        logger.info("Not found annotation file {}, load "
                    "wider-face categories.".format(anno_file))
        return widerfaceall_category_info(with_background)
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


def widerfaceall_category_info(with_background=True):
    """
    Get class id to category id map and category id
    to category name map of mixup wider_face dataset

    Args:
        with_background (bool, default True):
            whether load background as class 0.
    """
    label_map = widerface_label(with_background)
    label_map = sorted(label_map.items(), key=lambda x: x[1])
    cats = [l[0] for l in label_map]

    if with_background:
        cats.insert(0, 'background')

    clsid2catid = {i: i for i in range(len(cats))}
    catid2name = {i: name for i, name in enumerate(cats)}

    return clsid2catid, catid2name
