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

import numpy as np
from paddle.fluid.initializer import Uniform


# This file includes initializer, box encode, box decode
# initializer
def get_ctcn_conv_initializer(x, filter_size):
    c_in = x.shape[1]
    if isinstance(filter_size, int):
        fan_in = c_in * filter_size * filter_size
    else:
        fan_in = c_in * filter_size[0] * filter_size[1]
    std = np.sqrt(1.0 / fan_in)
    return Uniform(0. - std, std)


#box tools
def box_clamp1D(boxes, xmin, xmax):
    '''Clamp boxes.
    Args:
      boxes: (tensor) bounding boxes of (xmin,ymin,xmax,ymax), sized [N,2].
      xmin: (number) min value of x.
      xmax: (number) max value of x.
    '''
    np.clip(boxes[:, 0], xmin, xmax, out=boxes[:, 0])
    np.clip(boxes[:, 1], xmin, xmax, out=boxes[:, 1])
    return boxes


def box_iou1D(box1, box2):
    '''Compute the intersection over union of two set of boxes.

    The box order must be (xmin, xmax).

    Args:
      box1: (tensor) bounding boxes, sized [N,2].
      box2: (tensor) bounding boxes, sized [M,2].

    Return:
      (tensor) iou, sized [N,M].
    '''
    box1 = np.array(box1)
    box2 = np.array(box2)
    N = box1.shape[0]
    M = box2.shape[0]

    left = np.maximum(box1[:, None, 0], box2[:, 0])
    right = np.minimum(box1[:, None, 1], box2[:, 1])
    inter = (right - left).clip(min=0)
    area1 = np.abs(box1[:, 0] - box1[:, 1])
    area2 = np.abs(box2[:, 0] - box2[:, 1])
    iou = inter / (area1[:, None] + area2 - inter)
    return iou


def change_box_order(boxes, order):
    assert order in ['yy2yh', 'yh2yy']
    a = boxes[:, 0, None]
    b = boxes[:, 1, None]
    if order == 'yy2yh':
        return np.concatenate(((a + b) / 2, b - a), axis=1)
    return np.concatenate((a - b / 2, a + b / 2), axis=1)


def box_nms(bboxes, scores, threshold=0.5, mode='union'):
    '''Non maximum suppression.
    Args:
        bboxes: (tensor) bounding boxes, sized [N,2].
        scores: (tensor) confidence scores, sized [N,].
        threshold: (float) overlap threshold.
        mode: (str) 'union' or 'min'.

    Returns:
        keep: (tensor) selected indices.

    Reference:
        https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/nms/py_cpu_nms.py
    '''
    y1 = bboxes[:, 0]
    y2 = bboxes[:, 1]

    areas = (y2 - y1)
    order = np.argsort(-scores, axis=0)

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break

        yy1 = np.clip(y1[order[1:]], y1[i], None)
        yy2 = np.clip(y2[order[1:]], None, y2[i])
        h = np.clip(yy2 - yy1, 0, None)
        inter = h

        if mode == 'union':
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
        elif mode == 'min':
            ovr = inter / np.clip(areas[order[1:]], None, areas[i])
        else:
            raise TypeError('Unknown nms mode: %s.' % mode)

        ids = (ovr <= threshold).nonzero()[0]
        if ids.size == 0:
            break
        order = order[ids + 1]
    return np.array(keep, dtype='int64')


def soft_nms(props, method=0, sigma=1., Nt=0.7, threshold=0.001):
    '''
    param dets: dection results, 2 dims [N, 3]
    param props: predicted scores
    '''
    N = props.shape[0]

    for i in range(N):
        maxscore = props[i, 2]
        maxpos = i

        tx = props[i, 0]
        ty = props[i, 1]
        ts = props[i, 2]

        pos = i + 1
        while pos < N:
            if maxscore < props[pos, 2]:
                maxscore = props[pos, 2]
                maxpos = pos
            pos += 1

        props[i, 0] = props[maxpos, 0]
        props[i, 1] = props[maxpos, 1]
        props[i, 2] = props[maxpos, 2]

        props[maxpos, 0] = tx
        props[maxpos, 1] = ty
        props[maxpos, 2] = ts

        tx = props[i, 0]
        ty = props[i, 1]
        ts = props[i, 2]

        pos = i + 1
        while pos < N:
            x = props[pos, 0]
            y = props[pos, 1]
            s = props[pos, 2]

            max_begin = max(x, tx)
            min_end = min(y, ty)
            inter = max(0.0, min_end - max_begin)
            overlap = inter / (y - x + ty - tx - inter)

            if method == 1:
                if overlap > Nt:
                    weight = 1 - overlap
                else:
                    weight = 1
            elif method == 2:
                weight = np.exp(-(overlap**2) / sigma)
            else:
                if overlap > Nt:
                    weight = 0
                else:
                    weight = 1

            props[pos, 2] = weight * props[pos, 2]

            if props[pos, 2] < threshold:
                props[pos, 0] = props[N - 1, 0]
                props[pos, 1] = props[N - 1, 1]
                props[pos, 2] = props[N - 1, 2]
                N -= 1
                pos -= 1

            pos += 1
    keep = [i for i in range(N)]
    return props[keep]


# box encode and decode
class BoxCoder():
    def __init__(self):
        self.steps = (4, 8, 16, 32, 64, 128, 256, 512)
        self.fm_sizes = (128, 64, 32, 16, 8, 4, 2, 1)
        self.anchor_num = 3
        self.default_boxes = self._get_default_boxes()

    def _get_default_boxes(self):
        boxes = []
        for i, fm_size in enumerate(self.fm_sizes):
            for h in range(fm_size):
                cy = (h + 0.5) * self.steps[i]
                base_s = self.steps[i]
                boxes.append((cy, base_s))
                for p in range(self.anchor_num):
                    s = (base_s * 4.5 / 15.0) * (1.0 + p) / self.anchor_num
                    boxes.append((cy, base_s - s))
                    if base_s == 512:
                        step_s = (base_s * 4.5 / 15.0) / (2 * self.anchor_num)
                        boxes.append((cy, base_s - s - step_s))
                    else:
                        boxes.append((cy, base_s + s))
        return np.array(boxes)

    def encode(self, boxes, labels):
        def argmax(x):
            v = x.max(0)  # sort by cols, max_v, index
            i = np.argmax(x, 0)
            j = np.argmax(v, 0)  # v.max(0)[1][0]  # sort v, index
            return (i[j], j)  # return max index (row,col)

        labels = np.array(labels)
        default_boxes = self.default_boxes
        default_boxes = change_box_order(default_boxes, 'yh2yy')

        ious = box_iou1D(default_boxes, boxes)  # [#anchors, #obj]
        index = np.full(len(default_boxes), fill_value=-1, dtype='int64')

        masked_ious = ious.copy()

        while True:
            i, j = argmax(masked_ious)
            if masked_ious[i, j] < 1e-6:
                break
            index[i] = j
            masked_ious[i, :] = 0
            masked_ious[:, j] = 0

        mask = (index < 0) & (ious.max(1) >= 0.5)
        if mask.any():
            if np.squeeze(mask.nonzero()).size > 1:
                index[mask] = np.argmax(ious[np.squeeze(mask.nonzero())], 1)

        boxes = boxes[np.clip(index, a_min=0, a_max=None)]
        boxes = change_box_order(boxes, 'yy2yh')
        default_boxes = change_box_order(default_boxes, 'yy2yh')

        variances = (0.1, 0.2)
        loc_xy = (boxes[:, 0, None] - default_boxes[:, 0, None]
                  ) / default_boxes[:, 1, None] / variances[0]
        loc_wh = (
            boxes[:, 1, None] / default_boxes[:, 1, None] - 1.0) / variances[1]

        loc_targets = np.concatenate((loc_xy, loc_wh), axis=1)
        cls_targets = labels[index.clip(0, None)]
        cls_targets[index < 0] = 0

        return loc_targets, cls_targets

    def decode(self,
               loc_preds,
               cls_preds,
               score_thresh=0.6,
               nms_thresh=0.45,
               sigma_thresh=1.0,
               soft_thresh=0.01):
        '''Decode predicted loc/cls back to real box locations and class labels.

        Args:
          loc_preds: (tensor) predicted loc, sized [8732,2].
          cls_preds: (tensor) predicted conf, sized [8732,201].
          score_thresh: (float) threshold for object confidence score.
          nms_thresh: (float) threshold for box nms.

        Returns:
          boxes: (tensor) bbox locations, sized [#obj,2].
          labels: (tensor) class labels, sized [#obj,].
        '''

        variances = (0.1, 0.2)
        y = loc_preds[:, 0, None] * variances[
            0] * self.default_boxes[:, 1, None] + self.default_boxes[:, 0, None]
        h = (loc_preds[:, 1, None] * variances[1] + 1.0
             ) * self.default_boxes[:, 1, None]
        box_preds = np.concatenate((y - h / 2.0, y + h / 2.0), axis=1)

        boxes = []
        labels = []
        scores = []
        num_classes = cls_preds.shape[1]
        max_num = -1
        max_id = -1
        for i in range(num_classes - 1):
            score = cls_preds[:, i + 1]
            mask = score > score_thresh
            if not mask.any():
                continue
            box = box_preds[mask]
            score = score[mask]
            if len(score) > max_num:
                max_num = len(score)
                max_id = i

            keep = box_nms(box, score, nms_thresh)
            box = box[keep]
            score = score[keep]

            now_vector = np.concatenate((box, score[:, None]), axis=1)

            res = soft_nms(
                now_vector, method=2, sigma=sigma_thresh, threshold=soft_thresh)

            final_box = res[:, :2]
            final_score = res[:, 2]
            boxes.append(final_box)
            labels.append(np.full(len(final_box), fill_value=i, dtype='int64'))
            scores.append(final_score)
        if len(boxes) == 0:
            boxes.append(np.array([[0, 1.0]], dtype='float32'))
            labels.append(np.full(1, fill_value=1, dtype='int64'))
            scores.append(np.full(1, fill_value=1, dtype='float32'))
        boxes = np.concatenate(boxes, 0)
        labels = np.concatenate(labels, 0)
        scores = np.concatenate(scores, 0)
        return boxes, labels, scores
