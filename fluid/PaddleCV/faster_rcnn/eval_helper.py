#  Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
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

import os
import numpy as np
import paddle.fluid as fluid
import math
import box_utils
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from config import cfg


def box_decoder(deltas, boxes, weights):
    if boxes.shape[0] == 0:
        return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)

    boxes = boxes.astype(deltas.dtype, copy=False)

    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    wx, wy, ww, wh = weights
    dx = deltas[:, 0::4] * wx
    dy = deltas[:, 1::4] * wy
    dw = deltas[:, 2::4] * ww
    dh = deltas[:, 3::4] * wh

    # Prevent sending too large values into np.exp()
    dw = np.minimum(dw, cfg.bbox_clip)
    dh = np.minimum(dh, cfg.bbox_clip)

    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_w = np.exp(dw) * widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]

    pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
    # x1
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2 (note: "- 1" is correct; don't be fooled by the asymmetry)
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w - 1
    # y2 (note: "- 1" is correct; don't be fooled by the asymmetry)
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h - 1

    return pred_boxes


def clip_tiled_boxes(boxes, im_shape):
    """Clip boxes to image boundaries. im_shape is [height, width] and boxes
    has shape (N, 4 * num_tiled_boxes)."""
    assert boxes.shape[1] % 4 == 0, \
        'boxes.shape[1] is {:d}, but must be divisible by 4.'.format(
        boxes.shape[1]
    )
    # x1 >= 0
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
    return boxes


def get_nmsed_box(rpn_rois, confs, locs, class_nums, im_info,
                  numId_to_catId_map):
    lod = rpn_rois.lod()[0]
    rpn_rois_v = np.array(rpn_rois)
    variance_v = np.array(cfg.bbox_reg_weights)
    confs_v = np.array(confs)
    locs_v = np.array(locs)
    im_results = [[] for _ in range(len(lod) - 1)]
    new_lod = [0]
    for i in range(len(lod) - 1):
        start = lod[i]
        end = lod[i + 1]
        if start == end:
            continue
        locs_n = locs_v[start:end, :]
        rois_n = rpn_rois_v[start:end, :]
        rois_n = rois_n / im_info[i][2]
        rois_n = box_decoder(locs_n, rois_n, variance_v)
        rois_n = clip_tiled_boxes(rois_n, im_info[i][:2] / im_info[i][2])

        cls_boxes = [[] for _ in range(class_nums)]
        scores_n = confs_v[start:end, :]
        for j in range(1, class_nums):
            inds = np.where(scores_n[:, j] > cfg.TEST.score_thresh)[0]
            scores_j = scores_n[inds, j]
            rois_j = rois_n[inds, j * 4:(j + 1) * 4]
            dets_j = np.hstack((rois_j, scores_j[:, np.newaxis])).astype(
                np.float32, copy=False)
            keep = box_utils.nms(dets_j, cfg.TEST.nms_thresh)
            nms_dets = dets_j[keep, :]
            #add labels
            cat_id = numId_to_catId_map[j]
            label = np.array([cat_id for _ in range(len(keep))])
            nms_dets = np.hstack((nms_dets, label[:, np.newaxis])).astype(
                np.float32, copy=False)
            cls_boxes[j] = nms_dets
    # Limit to max_per_image detections **over all classes**
        image_scores = np.hstack(
            [cls_boxes[j][:, -2] for j in range(1, class_nums)])
        if len(image_scores) > cfg.TEST.detectiions_per_im:
            image_thresh = np.sort(image_scores)[-cfg.TEST.detectiions_per_im]
            for j in range(1, class_nums):
                keep = np.where(cls_boxes[j][:, -2] >= image_thresh)[0]
                cls_boxes[j] = cls_boxes[j][keep, :]

        im_results_n = np.vstack([cls_boxes[j] for j in range(1, class_nums)])
        im_results[i] = im_results_n
        new_lod.append(len(im_results_n) + new_lod[-1])
        boxes = im_results_n[:, :-2]
        scores = im_results_n[:, -2]
        labels = im_results_n[:, -1]
    im_results = np.vstack([im_results[k] for k in range(len(lod) - 1)])
    return new_lod, im_results


def get_dt_res(batch_size, lod, nmsed_out, data):
    dts_res = []
    nmsed_out_v = np.array(nmsed_out)
    assert (len(lod) == batch_size + 1), \
      "Error Lod Tensor offset dimension. Lod({}) vs. batch_size({})"\
                    .format(len(lod), batch_size)
    k = 0
    for i in range(batch_size):
        dt_num_this_img = lod[i + 1] - lod[i]
        image_id = int(data[i][-1])
        image_width = int(data[i][1][1])
        image_height = int(data[i][1][2])
        for j in range(dt_num_this_img):
            dt = nmsed_out_v[k]
            k = k + 1
            xmin, ymin, xmax, ymax, score, category_id = dt.tolist()
            w = xmax - xmin + 1
            h = ymax - ymin + 1
            bbox = [xmin, ymin, w, h]
            dt_res = {
                'image_id': image_id,
                'category_id': category_id,
                'bbox': bbox,
                'score': score
            }
            dts_res.append(dt_res)
    return dts_res


def draw_bounding_box_on_image(image_path, nms_out, draw_threshold, label_list):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size

    for dt in nms_out:
        xmin, ymin, xmax, ymax, score, category_id = dt.tolist()
        if score < draw_threshold:
            continue
        bbox = dt[:4]
        xmin, ymin, xmax, ymax = bbox
        draw.line(
            [(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin),
             (xmin, ymin)],
            width=4,
            fill='red')
        if image.mode == 'RGB':
            draw.text((xmin, ymin), label_list[int(category_id)], (255, 255, 0))
    image_name = image_path.split('/')[-1]
    print("image with bbox drawed saved as {}".format(image_name))
    image.save(image_name)
