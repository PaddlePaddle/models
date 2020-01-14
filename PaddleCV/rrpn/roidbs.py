# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved
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

import copy
import logging
import numpy as np
import os
import scipy.sparse
import random
import time
import matplotlib
import cv2
#import segm_utils
from config import cfg
from data_utils import DatasetPath
logger = logging.getLogger(__name__)


class ICDAR2015Dataset(object):
    """A class representing a ICDAR2015 dataset."""

    def __init__(self, mode):
        print('Creating: {}'.format(cfg.dataset))
        self.name = cfg.data_dir
        self.mode = mode
        data_path = DatasetPath(mode, self.name)
        data_dir = data_path.get_data_dir()
        file_list = data_path.get_file_list()
        self.image_dir = data_dir
        self.gt_dir = file_list

    def get_roidb(self):
        """Return an roidb corresponding to the txt dataset. Optionally:
           - include ground truth boxes in the roidb
        """
        image_list = os.listdir(self.image_dir)
        image_list.sort()
        im_infos = []
        count = 0
        for image in image_list:
            prefix = image[:-4]
            if image.split('.')[-1] != 'jpg':
                continue
            img_name = os.path.join(self.image_dir, image)
            gt_name = os.path.join(self.gt_dir, 'gt_' + prefix + '.txt')
            easy_boxes = []
            hard_boxes = []
            boxes = []
            gt_obj = open(gt_name, 'r', encoding='UTF-8-sig')
            gt_txt = gt_obj.read()
            gt_split = gt_txt.split('\n')
            img = cv2.imread(img_name)
            f = False
            for gt_line in gt_split:
                gt_ind = gt_line.split(',')

                # can get the text information
                if len(gt_ind) > 3 and '###' not in gt_ind[8]:
                    pt1 = (int(gt_ind[0]), int(gt_ind[1]))
                    pt2 = (int(gt_ind[2]), int(gt_ind[3]))
                    pt3 = (int(gt_ind[4]), int(gt_ind[5]))
                    pt4 = (int(gt_ind[6]), int(gt_ind[7]))
                    edge1 = np.sqrt((pt1[0] - pt2[0]) * (pt1[0] - pt2[0]) + (
                        pt1[1] - pt2[1]) * (pt1[1] - pt2[1]))
                    edge2 = np.sqrt((pt2[0] - pt3[0]) * (pt2[0] - pt3[0]) + (
                        pt2[1] - pt3[1]) * (pt2[1] - pt3[1]))
                    angle = 0
                    if edge1 > edge2:
                        width = edge1
                        height = edge2
                        if pt1[0] - pt2[0] != 0:
                            angle = -np.arctan(
                                float(pt1[1] - pt2[1]) /
                                float(pt1[0] - pt2[0])) / np.pi * 180
                        else:
                            angle = 90.0
                    elif edge2 >= edge1:
                        width = edge2
                        height = edge1
                        # print pt2[0], pt3[0]
                        if pt2[0] - pt3[0] != 0:
                            angle = -np.arctan(
                                float(pt2[1] - pt3[1]) /
                                float(pt2[0] - pt3[0])) / np.pi * 180
                        else:
                            angle = 90.0
                    if angle < -45.0:
                        angle = angle + 180
                    x_ctr = float(pt1[0] + pt3[
                        0]) / 2  # pt1[0] + np.abs(float(pt1[0] - pt3[0])) / 2
                    y_ctr = float(pt1[1] + pt3[
                        1]) / 2  # pt1[1] + np.abs(float(pt1[1] - pt3[1])) / 2
                    if self.mode == 'val':
                        easy_boxes.append(
                            list(np.array([pt1, pt2, pt3, pt4]).reshape(8)))
                    else:
                        easy_boxes.append([x_ctr, y_ctr, width, height, angle])
                # can‘t get the text information    
                if len(gt_ind) > 3 and '###' in gt_ind[8]:
                    pt1 = (int(gt_ind[0]), int(gt_ind[1]))
                    pt2 = (int(gt_ind[2]), int(gt_ind[3]))
                    pt3 = (int(gt_ind[4]), int(gt_ind[5]))
                    pt4 = (int(gt_ind[6]), int(gt_ind[7]))
                    edge1 = np.sqrt((pt1[0] - pt2[0]) * (pt1[0] - pt2[0]) + (
                        pt1[1] - pt2[1]) * (pt1[1] - pt2[1]))
                    edge2 = np.sqrt((pt2[0] - pt3[0]) * (pt2[0] - pt3[0]) + (
                        pt2[1] - pt3[1]) * (pt2[1] - pt3[1]))
                    angle = 0
                    if edge1 > edge2:
                        width = edge1
                        height = edge2
                        if pt1[0] - pt2[0] != 0:
                            angle = -np.arctan(
                                float(pt1[1] - pt2[1]) /
                                float(pt1[0] - pt2[0])) / np.pi * 180
                        else:
                            angle = 90.0
                    elif edge2 >= edge1:
                        width = edge2
                        height = edge1
                        if pt2[0] - pt3[0] != 0:
                            angle = -np.arctan(
                                float(pt2[1] - pt3[1]) /
                                float(pt2[0] - pt3[0])) / np.pi * 180
                        else:
                            angle = 90.0
                    if angle < -45.0:
                        angle = angle + 180
                    x_ctr = float(pt1[0] + pt3[
                        0]) / 2  # pt1[0] + np.abs(float(pt1[0] - pt3[0])) / 2
                    y_ctr = float(pt1[1] + pt3[
                        1]) / 2  # pt1[1] + np.abs(float(pt1[1] - pt3[1])) / 2
                    if self.mode == 'val':
                        hard_boxes.append(
                            list(np.array([pt1, pt2, pt3, pt4]).reshape(8)))
                    else:
                        hard_boxes.append([x_ctr, y_ctr, width, height, angle])

            #print(easy_boxes)
            if self.mode == 'train':
                boxes.extend(easy_boxes)
                # hard box only get 1/3 for train
                boxes.extend(hard_boxes[0:int(len(hard_boxes) / 3)])
                is_difficult = [0] * len(easy_boxes)
                is_difficult.extend([1] * int(len(hard_boxes) / 3))
            else:
                boxes.extend(easy_boxes)
                boxes.extend(hard_boxes)
                is_difficult = [0] * len(easy_boxes)
                is_difficult.extend([1] * int(len(hard_boxes)))
            len_of_bboxes = len(boxes)
            #is_difficult = [0] * len(easy_boxes)
            #is_difficult.extend([1] * int(len(hard_boxes)))
            is_difficult = np.array(is_difficult).reshape(
                1, len_of_bboxes).astype(np.int32)
            if self.mode == 'train':
                gt_boxes = np.zeros((len_of_bboxes, 5), dtype=np.int32)
            else:
                gt_boxes = np.zeros((len_of_bboxes, 8), dtype=np.int32)
            gt_classes = np.zeros((len_of_bboxes), dtype=np.int32)
            is_crowd = np.zeros((len_of_bboxes), dtype=np.int32)
            for idx in range(len(boxes)):
                if self.mode == 'train':
                    gt_boxes[idx, :] = [
                        boxes[idx][0], boxes[idx][1], boxes[idx][2],
                        boxes[idx][3], boxes[idx][4]
                    ]
                else:
                    gt_boxes[idx, :] = [
                        boxes[idx][0], boxes[idx][1], boxes[idx][2],
                        boxes[idx][3], boxes[idx][4], boxes[idx][5],
                        boxes[idx][6], boxes[idx][7]
                    ]
                gt_classes[idx] = 1
            if gt_boxes.shape[0] <= 0:
                continue
            gt_boxes = gt_boxes.astype(np.float64)
            im_info = {
                'im_id': count,
                'gt_classes': gt_classes,
                'image': img_name,
                'boxes': gt_boxes,
                'height': img.shape[0],
                'width': img.shape[1],
                'is_crowd': is_crowd,
                'is_difficult': is_difficult
            }
            im_infos.append(im_info)
            count += 1

        return im_infos


class ICDAR2017Dataset(object):
    """A class representing a ICDAR2017 dataset."""

    def __init__(self, mode):
        print('Creating: {}'.format(cfg.dataset))
        self.name = cfg.data_dir
        #print('**************', self.name)
        self.mode = mode
        data_path = DatasetPath(mode, self.name)
        data_dir = data_path.get_data_dir()
        #print("&**************", data_dir)
        file_list = data_path.get_file_list()
        self.image_dir = data_dir
        self.gt_dir = file_list

    def get_roidb(self):
        """Return an roidb corresponding to the json dataset. Optionally:
           - include ground truth boxes in the roidb
        """
        image_list = os.listdir(self.image_dir)
        image_list.sort()
        im_infos = []
        count = 0
        class_idx = 1
        class_name = {}
        post_fix = ['jpg', 'bmp', 'png']
        if self.mode == 'val':
            labels_map = get_labels_maps()
        for image in image_list:
            prefix = image[:-4]
            #print(image)

            if image.split('.')[-1] not in post_fix:
                continue
            img_name = os.path.join(self.image_dir, image)
            gt_name = os.path.join(self.gt_dir, 'gt_' + prefix + '.txt')
            gt_classes = []
            #boxes = []
            #hard_boxes = []
            boxes = []
            gt_obj = open(gt_name, 'r', encoding='UTF-8-sig')
            gt_txt = gt_obj.read()
            gt_split = gt_txt.split('\n')
            img = cv2.imread(img_name)
            f = False
            for gt_line in gt_split:
                gt_ind = gt_line.split(',')
                # can get the text information
                if len(gt_ind) > 3:
                    if self.mode == 'val':
                        gt_classes.append(labels_map[gt_ind[-1]])
                    else:
                        if gt_ind[-1] not in class_name:
                            class_name[gt_ind[-1]] = class_idx
                            #gt_classes.append(class_idx)
                            class_idx += 1
                        gt_classes.append(class_name[gt_ind[-1]])
                    pt1 = (int(gt_ind[0]), int(gt_ind[1]))
                    pt2 = (int(gt_ind[2]), int(gt_ind[3]))
                    pt3 = (int(gt_ind[4]), int(gt_ind[5]))
                    pt4 = (int(gt_ind[6]), int(gt_ind[7]))
                    edge1 = np.sqrt((pt1[0] - pt2[0]) * (pt1[0] - pt2[0]) + (
                        pt1[1] - pt2[1]) * (pt1[1] - pt2[1]))
                    edge2 = np.sqrt((pt2[0] - pt3[0]) * (pt2[0] - pt3[0]) + (
                        pt2[1] - pt3[1]) * (pt2[1] - pt3[1]))
                    angle = 0
                    if edge1 > edge2:
                        width = edge1
                        height = edge2
                        if pt1[0] - pt2[0] != 0:
                            angle = -np.arctan(
                                float(pt1[1] - pt2[1]) /
                                float(pt1[0] - pt2[0])) / np.pi * 180
                        else:
                            angle = 90.0
                    elif edge2 >= edge1:
                        width = edge2
                        height = edge1
                        # print pt2[0], pt3[0]
                        if pt2[0] - pt3[0] != 0:
                            angle = -np.arctan(
                                float(pt2[1] - pt3[1]) /
                                float(pt2[0] - pt3[0])) / np.pi * 180
                        else:
                            angle = 90.0
                    if angle < -45.0:
                        angle = angle + 180
                    x_ctr = float(pt1[0] + pt3[
                        0]) / 2  # pt1[0] + np.abs(float(pt1[0] - pt3[0])) / 2
                    y_ctr = float(pt1[1] + pt3[
                        1]) / 2  # pt1[1] + np.abs(float(pt1[1] - pt3[1])) / 2
                    if self.mode == 'val':
                        boxes.append(
                            list(np.array([pt1, pt2, pt3, pt4]).reshape(8)))
                    else:
                        boxes.append([x_ctr, y_ctr, width, height, angle])
            len_of_bboxes = len(boxes)
            #print(len_of_bboxes)
            is_difficult = np.zeros((len_of_bboxes, 1), dtype=np.int32)
            if self.mode == 'train':
                gt_boxes = np.zeros((len_of_bboxes, 5), dtype=np.int32)
            else:
                gt_boxes = np.zeros((len_of_bboxes, 8), dtype=np.int32)
            gt_classes = np.array(gt_classes).reshape(len_of_bboxes, 1)
            is_crowd = np.zeros((len_of_bboxes), dtype=np.int32)
            for idx in range(len(boxes)):
                if self.mode == 'train':
                    gt_boxes[idx, :] = [
                        boxes[idx][0], boxes[idx][1], boxes[idx][2],
                        boxes[idx][3], boxes[idx][4]
                    ]
                else:
                    gt_boxes[idx, :] = [
                        boxes[idx][0], boxes[idx][1], boxes[idx][2],
                        boxes[idx][3], boxes[idx][4], boxes[idx][5],
                        boxes[idx][6], boxes[idx][7]
                    ]
                #gt_classes[idx] = 1
            if gt_boxes.shape[0] <= 0:
                continue
            gt_boxes = gt_boxes.astype(np.float64)
            im_info = {
                'im_id': count,
                'gt_classes': gt_classes,
                'image': img_name,
                'boxes': gt_boxes,
                'height': img.shape[0],
                'width': img.shape[1],
                'is_crowd': is_crowd,
                'is_difficult': is_difficult
            }
            im_infos.append(im_info)
            count += 1
            if self.mode == 'train':
                with open(os.path.join(cfg.data_dir, 'label_list'), 'w') as g:
                    for k in class_name:
                        g.write(k + "\n")
        return im_infos


def get_labels_maps():
    labels_map = {}
    with open(os.path.join(cfg.data_dir, 'label_list')) as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            labels_map[line.strip()] = idx + 1
        return labels_map
