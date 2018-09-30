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


def box_decoder(target_box, prior_box, prior_box_var):
    proposals = np.zeros_like(target_box, dtype=np.float32)
    widths = prior_box[:, 2] - prior_box[:, 0] + 1.
    heights = prior_box[:, 3] - prior_box[:, 1] + 1.
    ctx = (prior_box[:, 2] + prior_box[:, 0]) / 2
    cty = (prior_box[:, 3] + prior_box[:, 1]) / 2
    dx = prior_box_var[0] * target_box[:, 0::4]
    dy = prior_box_var[1] * target_box[:, 1::4]
    dw = prior_box_var[2] * target_box[:, 2::4]
    dh = prior_box_var[3] * target_box[:, 3::4]

    dw = np.minimum(dw, EnvConfig.bbox_clip)
    dh = np.minimum(dh, EnvConfig.bbox_clip)

    pred_ctx = dx * widths[:, np.newaxis] + ctx[:, np.newaxis]
    pred_cty = dy * heights[:, np.newaxis] + cty[:, np.newaxis]
    pred_w = np.exp(dw) * widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]
    proposals[:, 0::4] = pred_ctx - pred_w / 2
    proposals[:, 1::4] = pred_cty - pred_h / 2
    proposals[:, 2::4] = pred_ctx + pred_w / 2 - 1
    proposals[:, 3::4] = pred_cty + pred_h / 2 - 1
    return proposals


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


def get_nmsed_box(args, rpn_rois, confs, locs, class_nums, im_info,
                  numId_to_catId_map):
    lod = rpn_rois.lod()[0]
    rpn_rois_v = np.array(rpn_rois)
    variance_v = np.array([0.1, 0.1, 0.2, 0.2])
    confs_v = np.array(confs)
    locs_v = np.array(locs)
    rois = box_decoder(locs_v, rpn_rois_v, variance_v)
    im_results = [[] for _ in range(len(lod) - 1)]
    new_lod = [0]
    for i in range(len(lod) - 1):
        start = lod[i]
        end = lod[i + 1]
        if start == end:
            continue
        rois_n = rois[start:end, :]
        rois_n = rois_n / im_info[i][2]
        rois_n = clip_tiled_boxes(rois_n, im_info[i][:2])

        cls_boxes = [[] for _ in range(class_nums)]
        scores_n = confs_v[start:end, :]
        for j in range(1, class_nums):
            inds = np.where(scores_n[:, j] > args.score_threshold)[0]
            scores_j = scores_n[inds, j]
            rois_j = rois_n[inds, j * 4:(j + 1) * 4]
            dets_j = np.hstack((rois_j, scores_j[:, np.newaxis])).astype(
                np.float32, copy=False)
            keep = box_utils.nms(dets_j, args.nms_threshold)
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
        if len(image_scores) > 100:
            image_thresh = np.sort(image_scores)[-100]
            for j in range(1, class_nums):
                keep = np.where(cls_boxes[j][:, -2] >= image_thresh)[0]
                cls_boxes[j] = cls_boxes[j][keep, :]

        im_results_n = np.vstack([cls_boxes[j] for j in range(1, class_nums)])
        im_results[i] = im_results_n
        new_lod.append(len(im_results_n) + new_lod[-1])
        boxes = im_results_n[:, :-2]
        scores = im_results_n[:, -2]
        labels = im_results_n[:, -1]
        #print('boxes after nms: {}, shape:{}'.format(np.sum(boxes),boxes.shape))
        #print('scores after nms: {}, shape:{}'.format(np.sum(scores),scores.shape))
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
