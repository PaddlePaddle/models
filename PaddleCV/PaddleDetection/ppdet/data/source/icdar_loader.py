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

import os
import numpy as np
import cv2


def transform_txt_line(gt_line, eval_mode=False):
    gt_ind = gt_line.split(',')
    pt1 = (int(gt_ind[0]), int(gt_ind[1]))
    pt2 = (int(gt_ind[2]), int(gt_ind[3]))
    pt3 = (int(gt_ind[4]), int(gt_ind[5]))
    pt4 = (int(gt_ind[6]), int(gt_ind[7]))
    if eval_mode == True:
        return [
            int(gt_ind[0]), int(gt_ind[1]), int(gt_ind[2]), int(gt_ind[3]),
            int(gt_ind[4]), int(gt_ind[5]), int(gt_ind[6]), int(gt_ind[7])
        ]
    edge1 = np.sqrt((pt1[0] - pt2[0]) * (pt1[0] - pt2[0]) + (pt1[1] - pt2[1]) *
                    (pt1[1] - pt2[1]))
    edge2 = np.sqrt((pt2[0] - pt3[0]) * (pt2[0] - pt3[0]) + (pt2[1] - pt3[1]) *
                    (pt2[1] - pt3[1]))
    angle = 0
    if edge1 > edge2:
        width = edge1
        height = edge2
        if pt1[0] - pt2[0] != 0:
            angle = -np.arctan(
                float(pt1[1] - pt2[1]) /
                float(pt1[0] - pt2[0])) / 3.1415926 * 180
        else:
            angle = 90.0
    elif edge2 >= edge1:
        width = edge2
        height = edge1
        if pt2[0] - pt3[0] != 0:
            angle = -np.arctan(
                float(pt2[1] - pt3[1]) /
                float(pt2[0] - pt3[0])) / 3.1415926 * 180
        else:
            angle = 90.0
    if angle < -45.0:
        angle = angle + 180
    x_ctr = float(pt1[0] + pt3[0]) / 2
    y_ctr = float(pt1[1] + pt3[1]) / 2
    gt_box = [x_ctr, y_ctr, width, height, angle]
    return gt_box


def load_roidb(anno_path,
               sample_num=-1,
               cname2cid=None,
               with_background=True,
               eval_mode=True):
    """
    Load ICDAR records with annotations in
    txt directory 'anno_path'

    Notes:
    ${anno_path} must contains xml file and image file path for annotations

    Args:
        @anno_path (str): root directory for icdar annotation data
        @sample_num (int): number of samples to load, -1 means all
        @with_background (bool): whether load background as a class.
                                 if True, total class number will
                                 be 81. default True

    Returns:
        (records, cname2clsid)
        'records' is list of dict whose structure is:
        {
            'im_file': im_fname, # image file name
            'im_id': im_id, # image id
            'h': im_h, # height of image
            'w': im_w, # width
            'gt_class': gt_class,
            'gt_bbox': gt_bbox,
        }
        'cname2id' is a dict to map category name to class id
    """
    data_dir = os.path.dirname(anno_path)
    # mapping category name to class id
    # if with_background is True:
    #   background:0, first_class:1, second_class:2, ...
    # if with_background is False:
    #   first_class:0, second_class:1, ...
    records = []
    ct = 0
    existence = False if cname2cid is None else True
    if cname2cid is None:
        cname2cid = {}
    with open(anno_path, 'r') as fr:
        while True:
            line = fr.readline()
            if not line:
                break
            img_file, txt_file = [os.path.join(data_dir, x) \
                    for x in line.strip().split()[:2]]
            if not os.path.isfile(txt_file):
                continue
            boxes = []
            gt_obj = open(txt_file, 'r', encoding='UTF-8-sig')
            gt_txt = gt_obj.read()
            gt_split = gt_txt.split('\n')
            print(img_file)
            img = cv2.imread(img_file)
            for gt_line in gt_split:
                gt_ind = gt_line.split(',')
                if len(gt_ind) == 1:
                    continue
                gt_box = transform_txt_line(gt_line)
                height = gt_box[3]
                width = gt_box[2]
                #if height * width < 32 * 32:
                #    continue
                #if height * width < 8 * 8:
                #    continue
                cname = gt_ind[8]
                if not existence and cname not in cname2cid:
                    # the background's id is 0, so need to add 1.
                    cname2cid[cname] = len(cname2cid) + int(with_background)
                elif existence and cname not in cname2cid:
                    raise KeyError(
                        'Not found cname[%s] in cname2cid when map it to cid.' %
                        (cname))
                gt_box.append(gt_ind[8])
                boxes.append(gt_box)
            len_of_bboxes = len(boxes)
            gt_boxes = []
            gt_classes = []
            for idx in range(len_of_bboxes):
                gt_classes.append(cname2cid[boxes[idx][5]])
                gt_boxes.append([
                    boxes[idx][0], boxes[idx][1], boxes[idx][2], boxes[idx][3],
                    boxes[idx][4]
                ])
            gt_classes = np.array(gt_classes).astype(np.int32)
            gt_boxes = np.array(gt_boxes).astype(np.float64)
            is_crowd = np.zeros(gt_classes.shape)
            im_info = {
                'im_file': img_file,
                'im_id': ct,
                'gt_class': gt_classes,
                'gt_bbox': gt_boxes,
                'h': img.shape[0],
                'w': img.shape[1],
                'is_crowd': is_crowd,
                'gt_poly': None
            }
            records.append(im_info)
            ct += 1
            if sample_num > 0 and ct >= sample_num:
                break

    assert len(records) > 0, 'not found any voc record in %s' % (anno_path)
    return [records, cname2cid]


def load(anno_path,
         sample_num=-1,
         use_default_label=True,
         multi_class=True,
         with_background=True,
         eval_mode=False):
    """
    Load ICDAR records with annotations in
    txt directory 'anno_path'

    Notes:
    ${anno_path} must contains xml file and image file path for annotations

    Args:
        @anno_path (str): root directory for icdar annotation data
        @sample_num (int): number of samples to load, -1 means all
        @multi_class (bool): whether differentiate multiple categories
        @with_background (bool): whether load background as a class.
                                 if True, total class number will
                                 be 81. default True

    Returns:
        (records, cname2clsid)
        'records' is list of dict whose structure is:
        {
            'im_file': im_fname, # image file name
            'im_id': im_id, # image id
            'h': im_h, # height of image
            'w': im_w, # width
            'gt_class': gt_class,
            'gt_bbox': gt_bbox,
        }
        'cname2id' is a dict to map category name to class id
    """
    data_dir = os.path.dirname(anno_path)

    # mapping category name to class id
    # if with_background is True:
    #   background:0, first_class:1, second_class:2, ...
    # if with_background is False:
    #   first_class:0, second_class:1, ...
    records = []
    ct = 0
    cname2cid = {}
    if multi_class:
        if not use_default_label:
            label_path = os.path.join(data_dir, 'label_list.txt')
            with open(label_path, 'r') as fr:
                label_id = int(with_background)
                for line in fr.readlines():
                    cname2cid[line.strip()] = label_id
                    label_id += 1
        else:
            cname2cid = icdar2017_label(with_background)
    else:
        cname2cid = {"has_object": 1}

    with open(anno_path, 'r') as fr:
        while True:
            line = fr.readline()
            if not line:
                break
            img_file, txt_file = [os.path.join(data_dir, x) \
                    for x in line.strip().split()[:2]]
            if not os.path.isfile(txt_file):
                continue
            boxes = []
            gt_obj = open(txt_file, 'r', encoding='UTF-8-sig')
            gt_txt = gt_obj.read()
            gt_split = gt_txt.split('\n')
            img = cv2.imread(img_file)
            if multi_class:
                for gt_line in gt_split:
                    gt_ind = gt_line.split(',')
                    if len(gt_ind) == 1:
                        continue
                    gt_box = transform_txt_line(gt_line, eval_mode=eval_mode)
                    height = gt_box[3]
                    width = gt_box[2]
                    if height * width < 32 * 32:
                        continue
                    #if height * width < 8 * 8:
                    #    print("###############################")
                    #    continue
                    gt_box.append(gt_ind[8])
                    boxes.append(gt_box)
                len_of_bboxes = len(boxes)
                gt_boxes = []
                gt_classes = []
                for idx in range(len_of_bboxes):
                    gt_classes.append(cname2cid[boxes[idx][8]])
                    gt_boxes.append([
                        boxes[idx][0], boxes[idx][1], boxes[idx][2],
                        boxes[idx][3], boxes[idx][4], boxes[idx][5],
                        boxes[idx][6], boxes[idx][7]
                    ])
                gt_classes = np.array(gt_classes).astype(np.int32)
                gt_boxes = np.array(gt_boxes).astype(np.float64)
                is_crowd = np.zeros(gt_classes.shape)
                is_difficult = np.zeros(gt_classes.shape)
                im_info = {
                    'im_file': img_file,
                    'im_id': ct,
                    'gt_class': gt_classes,
                    'gt_bbox': gt_boxes,
                    'h': img.shape[0],
                    'w': img.shape[1],
                    'is_crowd': is_crowd,
                    'gt_poly': None,
                    'is_difficult': is_difficult
                }
                records.append(im_info)
                ct += 1
                if sample_num > 0 and ct >= sample_num:
                    break
            else:
                easy_boxes = []
                hard_boxes = []
                for gt_line in gt_split:
                    gt_ind = gt_line.split(',')
                    if len(gt_ind) == 1:
                        continue
                    gt_box = transform_txt_line(gt_line, eval_mode=eval_mode)
                    if len(gt_ind) >= 8 and '###' not in gt_ind[-1]:
                        easy_boxes.append(gt_box)
                    if len(gt_ind) >= 8 and '###' in gt_ind[-1]:
                        hard_boxes.append(gt_box)
                boxes.extend(easy_boxes)
                if eval_mode == True:
                    boxes.extend(hard_boxes)
                    len_of_bboxes = len(boxes)
                    is_difficult = [0] * len(easy_boxes)
                    is_difficult.extend([1] * int(len(hard_boxes)))
                    gt_boxes = np.zeros((len_of_bboxes, 8), dtype=np.int32)

                else:
                    boxes.extend(hard_boxes[0:int(len(hard_boxes) / 3)])
                    len_of_bboxes = len(boxes)
                    is_difficult = [0] * len(easy_boxes)
                    is_difficult.extend([1] * int(len(hard_boxes) / 3))
                    gt_boxes = np.zeros((len_of_bboxes, 5), dtype=np.int32)
                is_difficult = np.array(is_difficult).reshape(
                    1, len_of_bboxes).astype(np.int32)
                #gt_boxes = np.zeros((len_of_bboxes, -1), dtype=np.int32)
                gt_classes = np.zeros((len_of_bboxes), dtype=np.int32)
                is_crowd = np.zeros((len_of_bboxes), dtype=np.int32)
                for idx in range(len(boxes)):
                    if eval_mode == True:
                        gt_boxes[idx, :] = [boxes[idx][0], boxes[idx][1], boxes[idx][2], boxes[idx][3], \
                                            boxes[idx][4], boxes[idx][5], boxes[idx][6], boxes[idx][7]]
                        gt_classes[idx] = 1
                    else:
                        gt_boxes[idx, :] = [
                            boxes[idx][0], boxes[idx][1], boxes[idx][2],
                            boxes[idx][3], boxes[idx][4]
                        ]
                        gt_classes[idx] = 1

                if gt_boxes.shape[0] <= 0:
                    continue
                gt_boxes = gt_boxes.astype(np.float64)
                im_info = {
                    'im_file': img_file,
                    'im_id': ct,
                    'gt_class': gt_classes,
                    'gt_bbox': gt_boxes,
                    'h': img.shape[0],
                    'w': img.shape[1],
                    'is_crowd': is_crowd,
                    'gt_poly': None,
                    'is_difficult': is_difficult
                }
                records.append(im_info)
                ct += 1
                if sample_num > 0 and ct >= sample_num:
                    break
    assert len(records) > 0, 'not found any voc record in %s' % (anno_path)
    return [records, cname2cid]


def icdar2017_label(with_background=True):
    labels_map = {
        'Arabic': 1,
        'English': 2,
        'Japanese': 3,
        'French': 4,
        'German': 5,
        'Chinese': 6,
        'Korean': 7,
        'Italian': 8,
        'Bangla': 9
    }
    if not with_background:
        labels_map = {k: v - 1 for k, v in labels_map.items()}
    return labels_map
