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
import os
import sys
sys.path.append('./so')
import time

import cv2
import numpy as np

from ConfigParser import ConfigParser
from PyCNNPredict import PyCNNPredict

#infer detector
def det_preprocessor(im, new_size, max_size):
    im = im.astype(np.float32, copy=False)
    img_mean = [0.485, 0.456, 0.406]
    img_std = [0.229, 0.224, 0.225]
    im = im[:, :, ::-1]
    im = im / 255
    im -= img_mean
    im /= img_std
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])  
    im_scale = float(new_size) / float(im_size_min)  
    # Prevent the biggest axis from being more than max_size
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
    channel_swap = (2, 0, 1)  #(batch, channel, height, width)
    im = im.transpose(channel_swap)
    return im, im_scale

def nms(dets, thresh):
    """nms"""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    dt_num = dets.shape[0]
    order = np.array(range(dt_num))

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return keep

def box_decoder(deltas, boxes, weights):
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
    clip_value = np.log(1000. / 16.)
    dw = np.minimum(dw, clip_value)
    dh = np.minimum(dh, clip_value)
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
    # x1 >= 0
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
    return boxes

def get_dt_res_common(rpn_rois_v, confs_v, locs_v, class_nums, im_info, im_id):
    dts_res = []
    if len(rpn_rois_v) == 0:
        return None
    variance_v = np.array([0.1, 0.1, 0.2, 0.2])
    img_height, img_width, img_scale = im_info
    tmp_v = box_decoder(locs_v, rpn_rois_v, variance_v)
    tmp_v = clip_tiled_boxes(tmp_v, [img_height, img_width])
    decoded_box_v = tmp_v / img_scale

    cls_boxes = [[] for _ in range(class_nums)]
    for j in range(1, class_nums):
        inds = np.where(confs_v[:, j] >= 0.1)[0]
        scores_j = confs_v[inds, j]
        rois_j = decoded_box_v[inds, j * 4:(j + 1) * 4]
        dets_j = np.hstack((rois_j, scores_j[:, np.newaxis])).astype(np.float32, copy=False)
        cls_rank = np.argsort(-dets_j[:, -1])
        dets_j = dets_j[cls_rank]
        keep = nms(dets_j, 0.5)
        nms_dets = dets_j[keep, :]
        cls_boxes[j] = nms_dets

    # Limit to max_per_image detections **over all classes**
    image_scores = np.hstack([cls_boxes[j][:, -1] for j in range(1, class_nums)])
    if len(image_scores) > 100:
        image_thresh = np.sort(image_scores)[-100]
        for j in range(1, class_nums):
            keep = np.where(cls_boxes[j][:, -1] >= image_thresh)[0]
            cls_boxes[j] = cls_boxes[j][keep, :]
    for j in range(1, class_nums):
        for dt in cls_boxes[j]:
            xmin, ymin, xmax, ymax, score = dt.tolist()
            w = xmax - xmin + 1
            h = ymax - ymin + 1
            bbox = [xmin, ymin, w, h]
            dt_res = {
                'image_id': im_id,
                'category_id': j,
                'bbox': bbox,
                'score': score
            }
            dts_res.append(dt_res)
    return dts_res

def test_det(img_path):
    conf_file = './conf/paddle-det.conf'
    prefix = 'paddle-classify_'
    conf = loadconfig(conf_file)
    det_prefix = 'paddle-det'
    class_nums = conf.getint(det_prefix, 'class_nums')
    new_size = conf.getfloat(det_prefix, 'new_size')
    max_size = conf.getfloat(det_prefix, 'max_size')
    predictor = PyCNNPredict()
    predictor.init(conf_file, prefix)
    im = cv2.imread(img_path)
    if im is None:
        print("image doesn't exist!")
        sys.exit(-1) 
    img_height_ori = im.shape[0]
    img_width_ori = im.shape[1]
    im, im_scale = det_preprocessor(im, new_size, max_size)
    im_height = np.round(img_height_ori * im_scale)
    im_width = np.round(img_width_ori * im_scale) 
    im_info = np.array([im_height, im_width, im_scale], dtype=np.float32)
    im_data_shape = np.array([1, im.shape[0], im.shape[1], im.shape[2]])
    im_info_shape = np.array([1, 3])
    im = im.flatten().astype(np.float32)
    im_info = im_info.flatten().astype(np.float32)
    inputdatas = [im, im_info]
    inputshapes = [im_data_shape.astype(np.int32), im_info_shape.astype(np.int32)]
    for ino in range(2):
        starttime = time.time() 
        res = predictor.predict(inputdatas, inputshapes, [])
        rpn_rois_v = res[0][0].reshape(-1, 4)
        confs_v = res[0][1].reshape(-1, class_nums)
        locs_v = res[0][2].reshape(-1, class_nums * 4)
        dts_res = get_dt_res_common(rpn_rois_v, confs_v, locs_v, class_nums, im_info, 0)
        print("Time:%.3f" % (time.time() - starttime))
    print(dts_res)

##infer cls 
def normwidth(size, margin=32):
    outsize = size // margin * margin
    return outsize


def loadconfig(configurefile):
    "load config from file"
    config = ConfigParser()
    config.readfp(open(configurefile, 'r'))
    return config


def resize_short(img, target_size):
    """ resize_short """
    percent = float(target_size) / min(img.shape[0], img.shape[1])
    resized_width = int(round(img.shape[1] * percent))
    resized_height = int(round(img.shape[0] * percent))

    resized_width = normwidth(resized_width)
    resized_height = normwidth(resized_height)
    resized = cv2.resize(img, (resized_width, resized_height))
    return resized


def crop_image(img, target_size, center):
    """ crop_image """
    height, width = img.shape[:2]
    size = target_size
    if center == True:
        w_start = (width - size) / 2
        h_start = (height - size) / 2
    else:
        w_start = random.randint(0, width - size)
        h_start = random.randint(0, height - size)
    w_end = w_start + size
    h_end = h_start + size
    img = img[h_start:h_end, w_start:w_end, :]
    return img


def cls_preprocessor(im, new_size):
    img_mean = [0.485, 0.456, 0.406]
    img_std = [0.229, 0.224, 0.225]

    img = resize_short(im, 224)
    img = crop_image(img, target_size=224, center=True)

    img = img[:, :, ::-1].astype('float32').transpose((2, 0, 1)) / 255
    img_mean = np.array(img_mean).reshape((3, 1, 1))
    img_std = np.array(img_std).reshape((3, 1, 1))
    img -= img_mean
    img /= img_std
    return img


def test_cls(img_path, model_name):
    conf_file = './conf/paddle-cls.conf'
    prefix = model_name + "_"
    conf = loadconfig(conf_file)
    predictor = PyCNNPredict()
    predictor.init(conf_file, prefix)
    im = cv2.imread(img_path)
    if im is None:
        print("image doesn't exist!")
        sys.exit(-1)
    im = cls_preprocessor(im, 224)
    im_data_shape = np.array([1, im.shape[0], im.shape[1], im.shape[2]])
    im = im.flatten().astype(np.float32)
    inputdatas = [im]
    inputshapes = [im_data_shape.astype(np.int32)]
    for ino in range(5):
        starttime = time.time()
        res = predictor.predict(inputdatas, inputshapes, [])
        print "Time:", time.time() - starttime 

    result = res[0][0]
    pred_label = np.argsort(result)[::-1][:1]
    
    print(pred_label)
    print(result[pred_label])

if __name__ == "__main__":
    if len(sys.argv)>1 :
        func = getattr(sys.modules[__name__], sys.argv[1])
        func(*sys.argv[2:])
    else:
        print >> sys.stderr,'tools.py command'
