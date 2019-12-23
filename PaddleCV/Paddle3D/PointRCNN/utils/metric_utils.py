#  Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import logging
import numpy as np
import utils.cyops.kitti_utils as kitti_utils 
from utils.config import cfg
from utils.box_utils import boxes_iou3d, box_nms_eval, boxes3d_to_bev
from utils.save_utils import save_rpn_feature, save_kitti_result, save_kitti_format

__all__ = ['calc_iou_recall', 'rpn_metric', 'rcnn_metric']

logging.root.handlers = []
FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)


def calc_iou_recall(rets, thresh_list):
    rpn_cls_label = rets['rpn_cls_label'][0]
    boxes3d = rets['rois'][0]
    seg_mask = rets['seg_mask'][0]
    sample_id = rets['sample_id'][0]
    gt_boxes3d = rets['gt_boxes3d'][0]
    gt_boxes3d_num = rets['gt_boxes3d'][1]

    gt_box_idx = 0
    recalled_bbox_list = [0] * len(thresh_list)
    gt_box_num = 0
    rpn_iou_sum = 0.
    for i in range(len(gt_boxes3d_num)):
        cur_rpn_cls_label = rpn_cls_label[i]
        cur_boxes3d = boxes3d[i]
        cur_seg_mask = seg_mask[i]
        cur_sample_id = sample_id[i]
        cur_gt_boxes3d = gt_boxes3d[gt_box_idx: gt_box_idx +
                                    gt_boxes3d_num[0][i]]
        gt_box_idx += gt_boxes3d_num[0][i]

        k = cur_gt_boxes3d.__len__() - 1
        while k >= 0 and np.sum(cur_gt_boxes3d[k]) == 0:
            k -= 1
        cur_gt_boxes3d = cur_gt_boxes3d[:k + 1]

        if cur_gt_boxes3d.shape[0] > 0:
            iou3d = boxes_iou3d(cur_boxes3d, cur_gt_boxes3d[:, 0:7])
            gt_max_iou = iou3d.max(axis=0)

            for idx, thresh in enumerate(thresh_list):
                recalled_bbox_list[idx] += np.sum(gt_max_iou > thresh)
            gt_box_num += cur_gt_boxes3d.__len__()

        fg_mask = cur_rpn_cls_label > 0
        correct = np.sum(np.logical_and(
            cur_seg_mask == cur_rpn_cls_label, fg_mask))
        union = np.sum(fg_mask) + np.sum(cur_seg_mask > 0) - correct
        rpn_iou = float(correct) / max(float(union), 1.0)
        rpn_iou_sum += rpn_iou
        logger.debug('sample_id:{}, rpn_iou:{}, gt_box_num:{}, recalled_bbox_list:{}'.format(
            sample_id, rpn_iou, gt_box_num, str(recalled_bbox_list)))

    return len(gt_boxes3d_num), gt_box_num, rpn_iou_sum, recalled_bbox_list


def rpn_metric(queue, mdict, lock, thresh_list, is_save_rpn_feature, kitti_feature_dir,
               seg_output_dir, kitti_output_dir, kitti_rcnn_reader, classes):
    while True:
        rets_dict = queue.get()
        if rets_dict is None:
            lock.acquire()
            mdict['exit_proc'] += 1
            lock.release()
            return 

        cnt, gt_box_num, rpn_iou_sum, recalled_bbox_list = calc_iou_recall(
            rets_dict, thresh_list)
        lock.acquire()
        mdict['total_cnt'] += cnt
        mdict['total_gt_bbox'] += gt_box_num
        mdict['total_rpn_iou'] += rpn_iou_sum
        for i, bbox_num in enumerate(recalled_bbox_list):
            mdict['total_recalled_bbox_list_{}'.format(i)] += bbox_num
        logger.debug("rpn_metric: {}".format(str(mdict)))
        lock.release()

        if is_save_rpn_feature:
            save_rpn_feature(rets_dict, kitti_feature_dir)
            save_kitti_result(
                rets_dict, seg_output_dir, kitti_output_dir, kitti_rcnn_reader, classes)


def rcnn_metric(queue, mdict, lock, thresh_list, kitti_rcnn_reader, roi_output_dir,
                refine_output_dir, final_output_dir, is_save_result=False):
    while True:
        rets_dict = queue.get()
        if rets_dict is None:
            lock.acquire()
            mdict['exit_proc'] += 1
            lock.release()
            return 
        
        for k,v in rets_dict.items():
            rets_dict[k] = v[0]

        rcnn_cls = rets_dict['rcnn_cls']
        rcnn_reg = rets_dict['rcnn_reg']
        roi_boxes3d = rets_dict['roi_boxes3d']
        roi_scores = rets_dict['roi_scores']

        # bounding box regression
        anchor_size = cfg.CLS_MEAN_SIZE[0]
        pred_boxes3d = kitti_utils.decode_bbox_target(
            roi_boxes3d, 
            rcnn_reg,
            anchor_size=np.array(anchor_size),
            loc_scope=cfg.RCNN.LOC_SCOPE,
            loc_bin_size=cfg.RCNN.LOC_BIN_SIZE,
            num_head_bin=cfg.RCNN.NUM_HEAD_BIN,
            get_xz_fine=True, 
            get_y_by_bin=cfg.RCNN.LOC_Y_BY_BIN,
            loc_y_scope=cfg.RCNN.LOC_Y_SCOPE,
            loc_y_bin_size=cfg.RCNN.LOC_Y_BIN_SIZE,
            get_ry_fine=True
        )

        # scoring
        if rcnn_cls.shape[1] == 1:
            raw_scores = rcnn_cls.reshape(-1)
            norm_scores = rets_dict['norm_scores']
            pred_classes = norm_scores > cfg.RCNN.SCORE_THRESH
            pred_classes = pred_classes.astype(np.float32)
        else:
            pred_classes = np.argmax(rcnn_cls, axis=1).reshape(-1)
            raw_scores = rcnn_cls[:, pred_classes]

        # evaluation
        gt_iou = rets_dict['gt_iou']
        gt_boxes3d = rets_dict['gt_boxes3d']
        
        # recall
        if gt_boxes3d.size > 0:
            gt_num = gt_boxes3d.shape[1]
            gt_boxes3d = gt_boxes3d.reshape((-1,7))
            iou3d = boxes_iou3d(pred_boxes3d, gt_boxes3d)
            gt_max_iou = iou3d.max(axis=0)
            refined_iou = iou3d.max(axis=1)

            recalled_num = (gt_max_iou > 0.7).sum()
            roi_boxes3d = roi_boxes3d.reshape((-1,7))
            iou3d_in = boxes_iou3d(roi_boxes3d, gt_boxes3d)
            gt_max_iou_in = iou3d_in.max(axis=0)

            lock.acquire()
            mdict['total_gt_bbox'] += gt_num
            for idx, thresh in enumerate(thresh_list):
                recalled_bbox_num = (gt_max_iou > thresh).sum() 
                mdict['total_recalled_bbox_list_{}'.format(idx)] += recalled_bbox_num
            for idx, thresh in enumerate(thresh_list):
                roi_recalled_bbox_num = (gt_max_iou_in > thresh).sum()
                mdict['total_roi_recalled_bbox_list_{}'.format(idx)] += roi_recalled_bbox_num 
            lock.release()
        
        # classification accuracy
        cls_label = gt_iou > cfg.RCNN.CLS_FG_THRESH
        cls_label = cls_label.astype(np.float32)
        cls_valid_mask = (gt_iou >= cfg.RCNN.CLS_FG_THRESH) | (gt_iou <= cfg.RCNN.CLS_BG_THRESH)
        cls_valid_mask = cls_valid_mask.astype(np.float32)
        cls_acc = (pred_classes == cls_label).astype(np.float32)
        cls_acc = (cls_acc * cls_valid_mask).sum() / max(cls_valid_mask.sum(), 1.0) * 1.0 
        
        iou_thresh = 0.7 if cfg.CLASSES == 'Car' else 0.5
        cls_label_refined = (gt_iou >= iou_thresh)
        cls_label_refined = cls_label_refined.astype(np.float32)
        cls_acc_refined = (pred_classes == cls_label_refined).astype(np.float32).sum() / max(cls_label_refined.shape[0], 1.0) 
        
        sample_id = rets_dict['sample_id']
        image_shape = kitti_rcnn_reader.get_image_shape(sample_id)
        
        if is_save_result:
            roi_boxes3d_np = roi_boxes3d
            pred_boxes3d_np = pred_boxes3d
            calib = kitti_rcnn_reader.get_calib(sample_id)
            save_kitti_format(sample_id, calib, roi_boxes3d_np, roi_output_dir, roi_scores, image_shape)
            save_kitti_format(sample_id, calib, pred_boxes3d_np, refine_output_dir, raw_scores, image_shape)
        
        inds = norm_scores > cfg.RCNN.SCORE_THRESH
        if inds.astype(np.float32).sum() == 0:
            logger.debug("The num of 'norm_scores > thresh' of sample {} is 0".format(sample_id))
            continue
        pred_boxes3d_selected = pred_boxes3d[inds]
        raw_scores_selected = raw_scores[inds]
        # NMS thresh
        boxes_bev_selected = boxes3d_to_bev(pred_boxes3d_selected)
        scores_selected, pred_boxes3d_selected = box_nms_eval(boxes_bev_selected, raw_scores_selected, pred_boxes3d_selected, cfg.RCNN.NMS_THRESH)
        calib = kitti_rcnn_reader.get_calib(sample_id)
        save_kitti_format(sample_id, calib, pred_boxes3d_selected, final_output_dir, scores_selected, image_shape)
        lock.acquire()
        mdict['total_det_num'] += pred_boxes3d_selected.shape[0]
        mdict['total_cls_acc'] += cls_acc
        mdict['total_cls_acc_refined'] += cls_acc_refined
        lock.release()
        logger.debug("rcnn_metric: {}".format(str(mdict)))

