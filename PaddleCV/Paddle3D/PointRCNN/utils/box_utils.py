#  Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.
"""
Contains proposal functions
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import paddle.fluid as fluid

from utils.config import cfg

__all__ = ["boxes3d_to_bev", "box_overlap_rotate", "boxes3d_to_bev", "box_iou", "box_nms"]


def boxes3d_to_bev(boxes3d):
    """
    Args:
        boxes3d: [N, 7], (x, y, z, h, w, l, ry)
    Return:
        boxes_bev: [N, 5], (x1, y1, x2, y2, ry)
    """
    boxes_bev = np.zeros((boxes3d.shape[0], 5), dtype='float32')

    cu, cv = boxes3d[:, 0], boxes3d[:, 2]
    half_l, half_w = boxes3d[:, 5] / 2, boxes3d[:, 4] / 2
    boxes_bev[:, 0], boxes_bev[:, 1] = cu - half_l, cv - half_w
    boxes_bev[:, 2], boxes_bev[:, 3] = cu + half_l, cv + half_w
    boxes_bev[:, 4] = boxes3d[:, 6]
    return boxes_bev


def rotate_around_center(center, angle_cos, angle_sin, corners):
    new_x = (corners[:, 0] - center[0]) * angle_cos + \
            (corners[:, 1] - center[1]) * angle_sin + center[0]
    new_y = -(corners[:, 0] - center[0]) * angle_sin + \
            (corners[:, 1] - center[1]) * angle_cos + center[1]
    return np.concatenate([new_x[:, np.newaxis], new_y[:, np.newaxis]], axis=-1)


def check_rect_cross(p1, p2, q1, q2):
    return min(p1[0], p2[0]) <= max(q1[0], q2[0]) and \
           min(q1[0], q2[0]) <= max(p1[0], p2[0]) and \
           min(p1[1], p2[1]) <= max(q1[1], q2[1]) and \
           min(q1[1], q2[1]) <= max(p1[1], p2[1])


def cross(p1, p2, p0):
    return (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (p1[1] - p0[1]);


def cross_area(a, b):
    return a[0] * b[1] - a[1] * b[0]


def intersection(p1, p0, q1, q0):
    if not check_rect_cross(p1, p0, q1, q0):
        return None

    s1 = cross(q0, p1, p0)
    s2 = cross(p1, q1, p0)
    s3 = cross(p0, q1, q0)
    s4 = cross(q1, p1, q0)
    if not (s1 * s2 > 0 and s3 * s4 > 0):
        return None

    s5 = cross(q1, p1, p0)
    if np.abs(s5 - s1) > 1e-8:
        return np.array([(s5 * q0[0] - s1 * q1[0]) / (s5 - s1),
                (s5 * q0[1] - s1 * q1[1]) / (s5 - s1)], dtype='float32')
    else:
        a0 = p0[1] - p1[1]
        b0 = p1[0] - p0[0]
        c0 = p0[0] * p1[1] - p1[0] * p0[1]
        a0 = q0[1] - q1[1]
        b0 = q1[0] - q0[0]
        c0 = q0[0] * q1[1] - q1[0] * q0[1]
        D = a0 * b1 - a1 * b0
        return np.array([(b0 * c1 - b1 * c0) / D, (a1 * c0 - a0 * c1) / D], dtype='float32')


def check_in_box2d(box, p):
    center_x = (box[0] + box[2]) / 2.
    center_y = (box[1] + box[3]) / 2.
    angle_cos = np.cos(-box[4])
    angle_sin = np.sin(-box[4])
    rot_x = (p[0] - center_x) * angle_cos + (p[1] - center_y) * angle_sin + center_x
    rot_y = -(p[0] - center_x) * angle_sin + (p[1] - center_y) * angle_cos + center_y
    return rot_x > box[0] - 1e-5 and rot_x < box[2] + 1e-5 and \
            rot_y > box[1] - 1e-5 and rot_y < box[3] + 1e-5


def point_cmp(a, b, center):
    return np.arctan2(a[1] - center[1], a[0] - center[0]) > \
            np.arctan2(b[1] - center[1], b[0] - center[0])


def box_overlap_rotate(cur_box, boxes):
    """
    Calculate box overlap with rotate, box: [x1, y1, x2, y2, angle]
    """
    areas = np.zeros((len(boxes), ), dtype='float32')
    cur_center = [(cur_box[0] + cur_box[2]) / 2., (cur_box[1] + cur_box[3]) / 2.]
    cur_corners = np.array([
            [cur_box[0], cur_box[1]], # (x1, y1)
            [cur_box[2], cur_box[1]], # (x2, y1)
            [cur_box[2], cur_box[3]], # (x2, y2)
            [cur_box[0], cur_box[3]], # (x1, y2)
            [cur_box[0], cur_box[1]], # (x1, y1)
            ], dtype='float32')
    cur_angle_cos = np.cos(cur_box[4])
    cur_angle_sin = np.sin(cur_box[4])
    cur_corners = rotate_around_center(cur_center, cur_angle_cos, cur_angle_sin, cur_corners)

    for i, box in enumerate(boxes):
        box_center = [(box[0] + box[2]) / 2., (box[1] + box[3]) / 2.]
        box_corners = np.array([
                [box[0], box[1]],
                [box[2], box[1]],
                [box[2], box[3]],
                [box[0], box[3]],
                [box[0], box[1]],
                ], dtype='float32')
        box_angle_cos = np.cos(box[4])
        box_angle_sin = np.sin(box[4])
        box_corners = rotate_around_center(box_center, box_angle_cos, box_angle_sin, box_corners)

        cross_points = np.zeros((16, 2), dtype='float32')
        cnt = 0
        # get intersection of lines
        for j in range(4):
            for k in range(4):
                inters = intersection(cur_corners[j + 1], cur_corners[j],
                                      box_corners[k + 1], box_corners[k])
                if inters is not None:
                    cross_points[cnt, :] = inters
                    cnt += 1
        # check corners
        for l in range(4):
            if check_in_box2d(cur_box, box_corners[l]):
                cross_points[cnt, :] = box_corners[l]
                cnt += 1
            if check_in_box2d(box, cur_corners[l]):
                cross_points[cnt, :] = cur_corners[l]
                cnt += 1

        if cnt > 0:
            poly_center = np.sum(cross_points[:cnt, :], axis=0) / cnt
        else:
            poly_center = np.zeros((2,))

        # sort the points of polygon
        for j in range(cnt - 1):
            for k in range(cnt - j - 1):
                if point_cmp(cross_points[k], cross_points[k + 1], poly_center):
                    cross_points[k], cross_points[k + 1] = \
                            cross_points[k + 1].copy(), cross_points[k].copy()

        # get the overlap areas
        area = 0.
        for j in range(cnt - 1):
            area += cross_area(cross_points[j] - cross_points[0],
                               cross_points[j + 1] - cross_points[0])
        areas[i] = np.abs(area) / 2.
    
    return areas


def box_iou(cur_box, boxes, box_type='normal'):
    cur_S = (cur_box[2] - cur_box[0]) * (cur_box[3] - cur_box[1])
    boxes_S = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    
    if box_type == 'normal':
        inter_x1 = np.maximum(cur_box[0], boxes[:, 0])
        inter_y1 = np.maximum(cur_box[1], boxes[:, 1])
        inter_x2 = np.minimum(cur_box[2], boxes[:, 2])
        inter_y2 = np.minimum(cur_box[3], boxes[:, 3])
        inter_w = np.maximum(inter_x2 - inter_x1, 0.)
        inter_h = np.maximum(inter_y2 - inter_y1, 0.)
        inter_area = inter_w * inter_h
    elif box_type == 'rotate':
        inter_area = box_overlap_rotate(cur_box, boxes)
    else:
        raise NotImplementedError

    return inter_area / np.maximum(cur_S + boxes_S - inter_area, 1e-8)


def box_nms(boxes, scores, proposals, thresh, topk, nms_type='normal'):
    assert nms_type in ['normal', 'rotate'], \
            "unknown nms type {}".format(nms_type)
    order = np.argsort(-scores)
    boxes = boxes[order]
    scores = scores[order]
    proposals = proposals[order]

    nmsed_scores = []
    nmsed_proposals = []
    cnt = 0
    while boxes.shape[0]:
        nmsed_scores.append(scores[0])
        nmsed_proposals.append(proposals[0])
        cnt +=1
        if cnt >= topk or boxes.shape[0] == 1:
            break
        iou = box_iou(boxes[0], boxes[1:], nms_type)
        boxes = boxes[1:][iou < thresh]
        scores = scores[1:][iou < thresh]
        proposals = proposals[1:][iou < thresh]
    return nmsed_scores, nmsed_proposals


def box_nms_eval(boxes, scores, proposals, thresh, nms_type='rotate'):
    assert nms_type in ['normal', 'rotate'], \
            "unknown nms type {}".format(nms_type)
    order = np.argsort(-scores)
    boxes = boxes[order]
    scores = scores[order]
    proposals = proposals[order]

    nmsed_scores = []
    nmsed_proposals = []
    while boxes.shape[0]:
        nmsed_scores.append(scores[0])
        nmsed_proposals.append(proposals[0])
        iou = box_iou(boxes[0], boxes[1:], nms_type)
        inds = iou < thresh
        boxes = boxes[1:][inds]
        scores = scores[1:][inds]
        proposals = proposals[1:][inds]
    nmsed_scores = np.asarray(nmsed_scores)
    nmsed_proposals = np.asarray(nmsed_proposals)
    return nmsed_scores, nmsed_proposals 

def boxes_iou3d(boxes1, boxes2):
    boxes1_bev = boxes3d_to_bev(boxes1)
    boxes2_bev = boxes3d_to_bev(boxes2)

    # bev overlap
    overlaps_bev = np.zeros((boxes1_bev.shape[0], boxes2_bev.shape[0]))
    for i in range(boxes1_bev.shape[0]):
        overlaps_bev[i, :] = box_overlap_rotate(boxes1_bev[i], boxes2_bev)

    # height overlap
    boxes1_height_min = (boxes1[:, 1] - boxes1[:, 3]).reshape(-1, 1)
    boxes1_height_max = boxes1[:, 1].reshape(-1, 1)
    boxes2_height_min = (boxes2[:, 1] - boxes2[:, 3]).reshape(1, -1)
    boxes2_height_max = boxes2[:, 1].reshape(1, -1)

    max_of_min = np.maximum(boxes1_height_min, boxes2_height_min)
    min_of_max = np.minimum(boxes1_height_max, boxes2_height_max)
    overlaps_h = np.maximum(min_of_max - max_of_min, 0.)

    # 3d iou
    overlaps_3d = overlaps_bev * overlaps_h

    vol_a = (boxes1[:, 3] * boxes1[:, 4] * boxes1[:, 5]).reshape(-1, 1)
    vol_b = (boxes2[:, 3] * boxes2[:, 4] * boxes2[:, 5]).reshape(1, -1)
    iou3d = overlaps_3d / np.maximum(vol_a + vol_b - overlaps_3d, 1e-7)

    return iou3d
