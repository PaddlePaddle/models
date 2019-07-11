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
import pycocotools.mask as mask_util
import six
from colormap import colormap
import cv2


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


def get_nmsed_box(rpn_rois, confs, locs, class_nums, im_info):
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
            dets_j = np.hstack((scores_j[:, np.newaxis], rois_j)).astype(
                np.float32, copy=False)
            keep = box_utils.nms(dets_j, cfg.TEST.nms_thresh)
            nms_dets = dets_j[keep, :]
            #add labels
            label = np.array([j for _ in range(len(keep))])
            nms_dets = np.hstack((nms_dets, label[:, np.newaxis])).astype(
                np.float32, copy=False)
            cls_boxes[j] = nms_dets
    # Limit to max_per_image detections **over all classes**
        image_scores = np.hstack(
            [cls_boxes[j][:, 1] for j in range(1, class_nums)])
        if len(image_scores) > cfg.TEST.detections_per_im:
            image_thresh = np.sort(image_scores)[-cfg.TEST.detections_per_im]
            for j in range(1, class_nums):
                keep = np.where(cls_boxes[j][:, 1] >= image_thresh)[0]
                cls_boxes[j] = cls_boxes[j][keep, :]

        im_results_n = np.vstack([cls_boxes[j] for j in range(1, class_nums)])
        im_results[i] = im_results_n
        new_lod.append(len(im_results_n) + new_lod[-1])
        boxes = im_results_n[:, 2:]
        scores = im_results_n[:, 1]
        labels = im_results_n[:, 0]
    im_results = np.vstack([im_results[k] for k in range(len(lod) - 1)])
    return new_lod, im_results


def get_dt_res(batch_size, lod, nmsed_out, data, num_id_to_cat_id_map):
    dts_res = []
    nmsed_out_v = np.array(nmsed_out)
    if nmsed_out_v.shape == (
            1,
            1, ):
        return dts_res
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
            num_id, score, xmin, ymin, xmax, ymax = dt.tolist()
            category_id = num_id_to_cat_id_map[num_id]
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


def get_segms_res(batch_size, lod, segms_out, data, num_id_to_cat_id_map):
    segms_res = []
    segms_out_v = np.array(segms_out)
    k = 0
    for i in range(batch_size):
        dt_num_this_img = lod[i + 1] - lod[i]
        image_id = int(data[i][-1])
        for j in range(dt_num_this_img):
            dt = segms_out_v[k]
            k = k + 1
            segm, num_id, score = dt.tolist()
            cat_id = num_id_to_cat_id_map[num_id]
            if six.PY3:
                if 'counts' in segm:
                    segm['counts'] = segm['counts'].decode("utf8")
            segm_res = {
                'image_id': image_id,
                'category_id': cat_id,
                'segmentation': segm,
                'score': score
            }
            segms_res.append(segm_res)
    return segms_res


def draw_bounding_box_on_image(image_path,
                               nms_out,
                               draw_threshold,
                               labels_map,
                               image=None):
    if image is None:
        image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size

    for dt in np.array(nms_out):
        num_id, score, xmin, ymin, xmax, ymax = dt.tolist()
        if score < draw_threshold:
            continue
        draw.line(
            [(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin),
             (xmin, ymin)],
            width=2,
            fill='red')
        if image.mode == 'RGB':
            draw.text((xmin, ymin), labels_map[num_id], (255, 255, 0))
    image_name = image_path.split('/')[-1]
    print("image with bbox drawed saved as {}".format(image_name))
    image.save(image_name)


def draw_mask_on_image(image_path, segms_out, draw_threshold, alpha=0.7):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    mask_color_id = 0
    w_ratio = .4
    image = np.array(image).astype('float32')
    for dt in np.array(segms_out):
        segm, num_id, score = dt.tolist()
        if score < draw_threshold:
            continue
        mask = mask_util.decode(segm) * 255
        color_list = colormap(rgb=True)
        color_mask = color_list[mask_color_id % len(color_list), 0:3]
        mask_color_id += 1
        for c in range(3):
            color_mask[c] = color_mask[c] * (1 - w_ratio) + w_ratio * 255
        idx = np.nonzero(mask)
        image[idx[0], idx[1], :] *= 1.0 - alpha
        image[idx[0], idx[1], :] += alpha * color_mask
    image = Image.fromarray(image.astype('uint8'))
    return image


def segm_results(im_results, masks, im_info):
    im_results = np.array(im_results)
    class_num = cfg.class_num
    M = cfg.resolution
    scale = (M + 2.0) / M
    lod = masks.lod()[0]
    masks_v = np.array(masks)
    boxes = im_results[:, 2:]
    labels = im_results[:, 0]
    segms_results = [[] for _ in range(len(lod) - 1)]
    sum = 0
    for i in range(len(lod) - 1):
        im_results_n = im_results[lod[i]:lod[i + 1]]
        cls_segms = []
        masks_n = masks_v[lod[i]:lod[i + 1]]
        boxes_n = boxes[lod[i]:lod[i + 1]]
        labels_n = labels[lod[i]:lod[i + 1]]
        im_h = int(round(im_info[i][0] / im_info[i][2]))
        im_w = int(round(im_info[i][1] / im_info[i][2]))
        boxes_n = box_utils.expand_boxes(boxes_n, scale)
        boxes_n = boxes_n.astype(np.int32)
        padded_mask = np.zeros((M + 2, M + 2), dtype=np.float32)
        for j in range(len(im_results_n)):
            class_id = int(labels_n[j])
            padded_mask[1:-1, 1:-1] = masks_n[j, class_id, :, :]

            ref_box = boxes_n[j, :]
            w = ref_box[2] - ref_box[0] + 1
            h = ref_box[3] - ref_box[1] + 1
            w = np.maximum(w, 1)
            h = np.maximum(h, 1)

            mask = cv2.resize(padded_mask, (w, h))
            mask = np.array(mask > cfg.mrcnn_thresh_binarize, dtype=np.uint8)
            im_mask = np.zeros((im_h, im_w), dtype=np.uint8)

            x_0 = max(ref_box[0], 0)
            x_1 = min(ref_box[2] + 1, im_w)
            y_0 = max(ref_box[1], 0)
            y_1 = min(ref_box[3] + 1, im_h)
            im_mask[y_0:y_1, x_0:x_1] = mask[(y_0 - ref_box[1]):(y_1 - ref_box[
                1]), (x_0 - ref_box[0]):(x_1 - ref_box[0])]
            sum += im_mask.sum()
            rle = mask_util.encode(
                np.array(
                    im_mask[:, :, np.newaxis], order='F'))[0]
            cls_segms.append(rle)
        segms_results[i] = np.array(cls_segms)[:, np.newaxis]
    segms_results = np.vstack([segms_results[k] for k in range(len(lod) - 1)])
    im_results = np.hstack([segms_results, im_results])
    return im_results[:, :3]


def coco17_labels():
    labels_map = {
        0: 'background',
        1: 'person',
        2: 'bicycle',
        3: 'car',
        4: 'motorcycle',
        5: 'airplane',
        6: 'bus',
        7: 'train',
        8: 'truck',
        9: 'boat',
        10: 'traffic light',
        11: 'fire hydrant',
        12: 'stop sign',
        13: 'parking meter',
        14: 'bench',
        15: 'bird',
        16: 'cat',
        17: 'dog',
        18: 'horse',
        19: 'sheep',
        20: 'cow',
        21: 'elephant',
        22: 'bear',
        23: 'zebra',
        24: 'giraffe',
        25: 'backpack',
        26: 'umbrella',
        27: 'handbag',
        28: 'tie',
        29: 'suitcase',
        30: 'frisbee',
        31: 'skis',
        32: 'snowboard',
        33: 'sports ball',
        34: 'kite',
        35: 'baseball bat',
        36: 'baseball glove',
        37: 'skateboard',
        38: 'surfboard',
        39: 'tennis racket',
        40: 'bottle',
        41: 'wine glass',
        42: 'cup',
        43: 'fork',
        44: 'knife',
        45: 'spoon',
        46: 'bowl',
        47: 'banana',
        48: 'apple',
        49: 'sandwich',
        50: 'orange',
        51: 'broccoli',
        52: 'carrot',
        53: 'hot dog',
        54: 'pizza',
        55: 'donut',
        56: 'cake',
        57: 'chair',
        58: 'couch',
        59: 'potted plant',
        60: 'bed',
        61: 'dining table',
        62: 'toilet',
        63: 'tv',
        64: 'laptop',
        65: 'mouse',
        66: 'remote',
        67: 'keyboard',
        68: 'cell phone',
        69: 'microwave',
        70: 'oven',
        71: 'toaster',
        72: 'sink',
        73: 'refrigerator',
        74: 'book',
        75: 'clock',
        76: 'vase',
        77: 'scissors',
        78: 'teddy bear',
        79: 'hair drier',
        80: 'toothbrush'
    }
    return labels_map
