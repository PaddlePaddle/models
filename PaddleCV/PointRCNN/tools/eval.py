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
import time
import shutil
import argparse
import ast
import logging
import multiprocessing
import numpy as np
from collections import OrderedDict 
import paddle
import paddle.fluid as fluid
from paddle.fluid.layers import control_flow
from paddle.fluid.contrib.extend_optimizer import extend_with_decoupled_weight_decay
import paddle.fluid.layers.learning_rate_scheduler as lr_scheduler

from models.point_rcnn import PointRCNN
from data.kitti_rcnn_reader import KittiRCNNReader
from utils.run_utils import check_gpu, parse_outputs, Stat
from utils.config import cfg, load_config
from utils import calibration as calib
import utils.cyops.kitti_utils as kitti_utils 
from utils.cyops.kitti_utils import rotate_pc_along_y_np
from utils.box_utils import boxes_iou3d, box_nms_eval, boxes3d_to_bev
import utils.calibration as calibration
#from tools.kitti_object_eval_python.evaluate import evaluate

logging.root.handlers = []
FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)

np.random.seed(1024)  # use same seed

# rpn_data_dir = "/home/ai/model/pytorch/PointRCNN/output/rpn/default/eval/epoch_200/val"
# rpn_data_dir = "/paddle/PointRCNN/output/rpn/eval/eval/epoch_200/val"
# rpn_data_dir = "./data/output/val"
# rpn_data_dir = "./data/val"
rpn_data_dir = "./data/train_aug_myfeature/val"


def parse_args():
    parser = argparse.ArgumentParser(
        "PointRCNN semantic segmentation train script")
    parser.add_argument(
        '--cfg',
        type=str,
        default='cfgs/default.yml',
        help='specify the config for training')
    parser.add_argument(
        '--use_gpu',
        type=ast.literal_eval,
        default=True,
        help='default use gpu.')
    parser.add_argument(
        '--eval_mode',
        type=str,
        default='rpn',
        help='specify the training mode')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help='evaluation batch size, default 1')
    parser.add_argument(
        '--ckpt_dir',
        type=str,
        default='test',
        help='specify a ckpt directory to be evaluated if needed')
    parser.add_argument(
        '--output_dir',
        type=str,
        default='output',
        help='output directory')
    parser.add_argument(
        '--save_rpn_feature',
        action='store_true',
        default=False,
        help='save features for separately rcnn training and evaluation')
    parser.add_argument(
        '--save_result',
        action='store_true',
        default=False,
        help='save roi and refine result of evaluation')
    parser.add_argument(
        '--rcnn_eval_roi_dir',
        type=str,
        default=rpn_data_dir+"/detections/data",  # None,
        help='specify the saved rois for rcnn evaluation when using rcnn_offline mode')
    parser.add_argument(
        '--rcnn_eval_feature_dir',
        type=str,
        default=rpn_data_dir + "/features",  # None,
        help='specify the saved features for rcnn evaluation when using rcnn_offline mode')
    parser.add_argument(
        '--log_interval',
        type=int,
        default=1,
        help='mini-batch interval to log.')
    args = parser.parse_args()
    return args


def save_rpn_feature(rets, kitti_features_dir):
    """
    save rpn features for RCNN training
    """

    sample_id = rets['sample_id'][0]
    backbone_xyz = rets['backbone_xyz'][0]
    backbone_feature = rets['backbone_feature'][0]
    pts_features = rets['pts_features'][0]
    seg_mask = rets['seg_mask'][0]
    rpn_cls = rets['rpn_cls'][0]

    for i in range(len(sample_id)):
        pts_intensity = pts_features[i, :, 0]
        s_id = sample_id[i, 0]

        output_file = os.path.join(kitti_features_dir, '%06d.npy' % s_id)
        xyz_file = os.path.join(kitti_features_dir, '%06d_xyz.npy' % s_id)
        seg_file = os.path.join(kitti_features_dir, '%06d_seg.npy' % s_id)
        intensity_file = os.path.join(
            kitti_features_dir, '%06d_intensity.npy' % s_id)
        np.save(output_file, backbone_feature[i])
        np.save(xyz_file, backbone_xyz[i])
        np.save(seg_file, seg_mask[i])
        np.save(intensity_file, pts_intensity)
        rpn_scores_raw_file = os.path.join(
            kitti_features_dir, '%06d_rawscore.npy' % s_id)
        np.save(rpn_scores_raw_file, rpn_cls[i])


def save_kitti_result(rets, seg_output_dir, kitti_output_dir, reader, classes):
    sample_id = rets['sample_id'][0]
    roi_scores_row = rets['roi_scores_row'][0]
    bboxes3d = rets['rois'][0]
    pts_rect = rets['pts_rect'][0]
    seg_mask = rets['seg_mask'][0]
    rpn_cls_label = rets['rpn_cls_label'][0]
    gt_boxes3d = rets['gt_boxes3d'][0]
    gt_boxes3d_num = rets['gt_boxes3d'][1]

    for i in range(len(sample_id)):
        s_id = sample_id[i, 0]

        seg_result_data = np.concatenate((pts_rect[i].reshape(-1, 3),
                                          rpn_cls_label[i].reshape(-1, 1),
                                          seg_mask[i].reshape(-1, 1)),
                                         axis=1).astype('float16')
        seg_output_file = os.path.join(seg_output_dir, '%06d.npy' % s_id)
        np.save(seg_output_file, seg_result_data)

        scores = roi_scores_row[i, :]
        bbox3d = bboxes3d[i, :]
        img_shape = reader.get_image_shape(s_id)
        calib = reader.get_calib(s_id)

        corners3d = kitti_utils.boxes3d_to_corners3d(bbox3d)
        img_boxes, _ = calib.corners3d_to_img_boxes(corners3d)

        img_boxes[:, 0] = np.clip(img_boxes[:, 0], 0, img_shape[1] - 1)
        img_boxes[:, 1] = np.clip(img_boxes[:, 1], 0, img_shape[0] - 1)
        img_boxes[:, 2] = np.clip(img_boxes[:, 2], 0, img_shape[1] - 1)
        img_boxes[:, 3] = np.clip(img_boxes[:, 3], 0, img_shape[0] - 1)

        img_boxes_w = img_boxes[:, 2] - img_boxes[:, 0]
        img_boxes_h = img_boxes[:, 3] - img_boxes[:, 1]
        box_valid_mask = np.logical_and(
            img_boxes_w < img_shape[1] * 0.8, img_boxes_h < img_shape[0] * 0.8)

        kitti_output_file = os.path.join(kitti_output_dir, '%06d.txt' % s_id)
        with open(kitti_output_file, 'w') as f:
            for k in range(bbox3d.shape[0]):
                if box_valid_mask[k] == 0:
                    continue
                x, z, ry = bbox3d[k, 0], bbox3d[k, 2], bbox3d[k, 6]
                beta = np.arctan2(z, x)
                alpha = -np.sign(beta) * np.pi / 2 + beta + ry

                f.write('{} -1 -1 {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}\n'.format(
                    classes, alpha, img_boxes[k, 0], img_boxes[k, 1], img_boxes[k, 2], img_boxes[k, 3],
                    bbox3d[k, 3], bbox3d[k, 4], bbox3d[k, 5], bbox3d[k, 0], bbox3d[k, 1], bbox3d[k, 2],
                    bbox3d[k, 6], scores[k]))


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


def save_kitti_format(sample_id, calib, bbox3d, kitti_output_dir, scores, img_shape):
    corners3d = kitti_utils.boxes3d_to_corners3d(bbox3d)
    img_boxes, _ = calib.corners3d_to_img_boxes(corners3d)
    img_boxes[:, 0] = np.clip(img_boxes[:, 0], 0, img_shape[1] - 1)
    img_boxes[:, 1] = np.clip(img_boxes[:, 1], 0, img_shape[0] - 1)
    img_boxes[:, 2] = np.clip(img_boxes[:, 2], 0, img_shape[1] - 1)
    img_boxes[:, 3] = np.clip(img_boxes[:, 3], 0, img_shape[0] - 1)

    img_boxes_w = img_boxes[:, 2] - img_boxes[:, 0]
    img_boxes_h = img_boxes[:, 3] - img_boxes[:, 1]
    box_valid_mask = np.logical_and(img_boxes_w < img_shape[1] * 0.8, img_boxes_h < img_shape[0] * 0.8)

    kitti_output_file = os.path.join(kitti_output_dir, '%06d.txt' % sample_id)
    with open(kitti_output_file, 'w') as f:
        for k in range(bbox3d.shape[0]):
            if box_valid_mask[k] == 0:
                continue
            x, z, ry = bbox3d[k, 0], bbox3d[k, 2], bbox3d[k, 6]
            beta = np.arctan2(z, x)
            alpha = -np.sign(beta) * np.pi / 2 + beta + ry

            f.write('%s -1 -1 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n' %
                  (cfg.CLASSES, alpha, img_boxes[k, 0], img_boxes[k, 1], img_boxes[k, 2], img_boxes[k, 3],
                   bbox3d[k, 3], bbox3d[k, 4], bbox3d[k, 5], bbox3d[k, 0], bbox3d[k, 1], bbox3d[k, 2],
                   bbox3d[k, 6], scores[k]))

def rpn_metric(queue, mdict, thresh_list, is_save_rpn_feature, kitti_feature_dir,
               seg_output_dir, kitti_output_dir, kitti_rcnn_reader, classes):
    while True:
        rets_dict = queue.get()
        if rets_dict is None:
            return 

        cnt, gt_box_num, rpn_iou_sum, recalled_bbox_list = calc_iou_recall(
            rets_dict, thresh_list)
        mdict['total_cnt'] += cnt
        mdict['total_gt_bbox'] += gt_box_num
        mdict['total_rpn_iou'] += rpn_iou_sum
        for i, bbox_num in enumerate(recalled_bbox_list):
            mdict['total_recalled_bbox_list_{}'.format(i)] += bbox_num
        logger.debug("rpn_metric: {}".format(str(mdict)))

        if is_save_rpn_feature:
            save_rpn_feature(rets_dict, kitti_feature_dir)
            save_kitti_result(
                rets_dict, seg_output_dir, kitti_output_dir, kitti_rcnn_reader, classes)


def rcnn_metric(queue, mdict, thresh_list, kitti_rcnn_reader, roi_output_dir, refine_output_dir, final_output_dir, is_save_result=False):
    while True:
        rets_dict = queue.get()
        if rets_dict is None:
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
        gt_num = gt_boxes3d.shape[1]
        if gt_num > 0:
            gt_boxes3d = gt_boxes3d.reshape((-1,7))
            iou3d = boxes_iou3d(pred_boxes3d, gt_boxes3d)
            gt_max_iou = iou3d.max(axis=0)
            refined_iou = iou3d.max(axis=1)

            for idx, thresh in enumerate(thresh_list):
                recalled_bbox_num = (gt_max_iou > thresh).sum() 
                mdict['total_recalled_bbox_list_{}'.format(idx)] += recalled_bbox_num

            recalled_num = (gt_max_iou > 0.7).sum()
            mdict['total_gt_bbox'] += gt_num
            roi_boxes3d = roi_boxes3d.reshape((-1,7))
            iou3d_in = boxes_iou3d(roi_boxes3d, gt_boxes3d)
            gt_max_iou_in = iou3d_in.max(axis=0)

            for idx, thresh in enumerate(thresh_list):
                roi_recalled_bbox_num = (gt_max_iou_in > thresh).sum()
                mdict['total_roi_recalled_bbox_list_{}'.format(idx)] += roi_recalled_bbox_num 
        
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
        
        mdict['total_cls_acc'] += cls_acc
        mdict['total_cls_acc_refined'] += cls_acc_refined
            
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
        mdict['total_det_num'] += pred_boxes3d_selected.shape[0]
        save_kitti_format(sample_id, calib, pred_boxes3d_selected, final_output_dir, scores_selected, image_shape)
        logger.debug("rcnn_metric: {}".format(str(mdict)))

def eval():
    args = parse_args()
    # check whether the installed paddle is compiled with GPU
    check_gpu(args.use_gpu)

    load_config(args.cfg)

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    if args.eval_mode == 'rpn':
        cfg.RPN.ENABLED = True
        cfg.RCNN.ENABLED = False
    elif args.eval_mode == 'rcnn':
        cfg.RCNN.ENABLED = True
        cfg.RPN.ENABLED = cfg.RPN.FIXED = True
    elif args.eval_mode == 'rcnn_offline':
        cfg.RCNN.ENABLED = True
        cfg.RPN.ENABLED = False
    else:
        raise NotImplementedError

    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)

    # build model
    startup = fluid.Program()
    eval_prog = fluid.Program()
    with fluid.program_guard(eval_prog, startup):
        with fluid.unique_name.guard():
            eval_model = PointRCNN(cfg, args.batch_size, True, 'TEST')
            eval_model.build()
            eval_pyreader = eval_model.get_pyreader()
            eval_feeds = eval_model.get_feeds()
            eval_outputs = eval_model.get_outputs()
    eval_prog = eval_prog.clone(True)

    extra_keys = []
    if args.eval_mode == 'rpn':
        extra_keys.extend(['sample_id', 'rpn_cls_label', 'gt_boxes3d'])
        if args.save_rpn_feature:
            extra_keys.extend(['pts_rect', 'pts_features', 'pts_input',])
    eval_keys, eval_values = parse_outputs(
        eval_outputs, prog=eval_prog, extra_keys=extra_keys)

    eval_compile_prog = fluid.compiler.CompiledProgram(
        eval_prog).with_data_parallel()

    exe.run(startup)

    # load checkpoint
    assert os.path.isdir(
        args.ckpt_dir), "ckpt_dir {} not a directory".format(args.ckpt_dir)

    def if_exist(var):
        return os.path.exists(os.path.join(args.ckpt_dir, var.name))
    fluid.io.load_vars(exe, args.ckpt_dir, eval_prog, predicate=if_exist)

    kitti_feature_dir = os.path.join(args.output_dir, 'features')
    kitti_output_dir = os.path.join(args.output_dir, 'detections', 'data')
    seg_output_dir = os.path.join(args.output_dir, 'seg_result')
    if args.save_rpn_feature:
        if os.path.exists(kitti_feature_dir):
            shutil.rmtree(kitti_feature_dir)
        os.makedirs(kitti_feature_dir)
        if os.path.exists(kitti_output_dir):
            shutil.rmtree(kitti_output_dir)
        os.makedirs(kitti_output_dir)
        if os.path.exists(seg_output_dir):
            shutil.rmtree(seg_output_dir)
        os.makedirs(seg_output_dir)

    # must make sure these dirs existing 
    roi_output_dir = os.path.join('./result_dir', 'roi_result', 'data')
    refine_output_dir = os.path.join('./result_dir', 'refine_result', 'data')
    final_output_dir = os.path.join("./result_dir", 'final_result', 'data')
    if not os.path.exists(final_output_dir):
        os.makedirs(final_output_dir)
    if args.save_result:
        if not os.path.exists(roi_output_dir):
            os.makedirs(roi_output_dir)
        if not os.path.exists(refine_output_dir):
            os.makedirs(refine_output_dir)

    # get reader
    kitti_rcnn_reader = KittiRCNNReader(data_dir='./data',
                                        npoints=cfg.RPN.NUM_POINTS,
                                        split=cfg.TEST.SPLIT,
                                        mode='EVAL',
                                        classes=cfg.CLASSES,
                                        rcnn_eval_roi_dir=args.rcnn_eval_roi_dir,
                                        rcnn_eval_feature_dir=args.rcnn_eval_feature_dir)
    eval_reader = kitti_rcnn_reader.get_multiprocess_reader(args.batch_size, eval_feeds)
    eval_pyreader.decorate_sample_list_generator(eval_reader, place)

    thresh_list = [0.1, 0.3, 0.5, 0.7, 0.9]
    queue = multiprocessing.Queue(128)
    mgr = multiprocessing.Manager()
    mdict = mgr.dict()
    if cfg.RPN.ENABLED:
        mdict['total_gt_bbox'] = 0
        mdict['total_cnt'] = 0
        mdict['total_rpn_iou'] = 0
        for i in range(len(thresh_list)):
            mdict['total_recalled_bbox_list_{}'.format(i)] = 0

        p_list = []
        for i in range(4):
            p_list.append(multiprocessing.Process(
                target=rpn_metric,
                args=(queue, mdict, thresh_list, args.save_rpn_feature, kitti_feature_dir,
                      seg_output_dir, kitti_output_dir, kitti_rcnn_reader, cfg.CLASSES)))
            p_list[-1].start()
    
    if cfg.RCNN.ENABLED:
        for i in range(len(thresh_list)):
            mdict['total_recalled_bbox_list_{}'.format(i)] = 0
            mdict['total_roi_recalled_bbox_list_{}'.format(i)] = 0
        mdict['total_cls_acc'] = 0 
        mdict['total_cls_acc_refined'] = 0
        mdict['total_det_num'] = 0
        mdict['total_gt_bbox'] = 0
        p_list = []
        for i in range(4):
            p_list.append(multiprocessing.Process(
                target=rcnn_metric,
                args=(queue, mdict, thresh_list, kitti_rcnn_reader, roi_output_dir,
                      refine_output_dir, final_output_dir, args.save_result)
            ))
            p_list[-1].start()

    try:
        eval_pyreader.start()
        eval_iter = 0
        start_time = time.time()
        
        cur_time = time.time()
        while True:
            eval_outs = exe.run(eval_compile_prog, fetch_list=eval_values, return_numpy=False)
            rets_dict = {k: (np.array(v), v.recursive_sequence_lengths()) 
                    for k, v in zip(eval_keys, eval_outs)}
            run_time = time.time() - cur_time
            cur_time = time.time()
            queue.put(rets_dict)
            eval_iter += 1

            logger.info("[EVAL] iter {}, time: {:.2f}".format(
                eval_iter, run_time))

    except fluid.core.EOFException:
        # terminate metric process
        for i in range(len(p_list)):
            queue.put(None)
        for p in p_list:
            if p.is_alive():
                p.join()

        end_time = time.time()
        logger.info("[EVAL] total {} iter finished, average time: {:.2f}".format(
            eval_iter, (end_time - start_time) / float(eval_iter)))

        if cfg.RPN.ENABLED:
            avg_rpn_iou = mdict['total_rpn_iou'] / max(len(kitti_rcnn_reader), 1.)
            logger.info("average rpn iou: {:.3f}".format(avg_rpn_iou))
            total_gt_bbox = float(max(mdict['total_gt_bbox'], 1.0))
            for idx, thresh in enumerate(thresh_list):
                recall = mdict['total_recalled_bbox_list_{}'.format(idx)] / total_gt_bbox
                logger.info("total bbox recall(thresh={:.3f}): {} / {} = {:.3f}".format(
                    thresh, mdict['total_recalled_bbox_list_{}'.format(idx)], mdict['total_gt_bbox'], recall))

        if cfg.RCNN.ENABLED:
            cnt = float(max(eval_iter, 1.0))
            avg_cls_acc = mdict['total_cls_acc'] / cnt
            avg_cls_acc_refined = mdict['total_cls_acc_refined'] / cnt
            avg_det_num = mdict['total_det_num'] / cnt
            
            logger.info("avg_cls_acc: {}".format(avg_cls_acc))
            logger.info("avg_cls_acc_refined: {}".format(avg_cls_acc_refined))
            logger.info("avg_det_num: {}".format(avg_det_num))             
            
            total_gt_bbox = float(max(mdict['total_gt_bbox'], 1.0))
            for idx, thresh in enumerate(thresh_list):
                cur_roi_recall = mdict['total_roi_recalled_bbox_list_{}'.format(idx)] / total_gt_bbox
                logger.info('total roi bbox recall(thresh=%.3f): %d / %d = %f' % (
                    thresh, mdict['total_roi_recalled_bbox_list_{}'.format(idx)], total_gt_bbox, cur_roi_recall))
            
            for idx, thresh in enumerate(thresh_list):
                cur_recall = mdict['total_recalled_bbox_list_{}'.format(idx)] / total_gt_bbox
                logger.info('total bbox recall(thresh=%.2f) %d / %.2f = %.4f' % (
                    thresh, mdict['total_recalled_bbox_list_{}'.format(idx)], total_gt_bbox, cur_recall))
            
            split_file = os.path.join('./data/KITTI', 'ImageSets', 'val.txt')
            image_idx_list = [x.strip() for x in open(split_file).readlines()]
            empty_cnt = 0
            for k in range(image_idx_list.__len__()):
                cur_file = os.path.join(final_output_dir, '%s.txt' % image_idx_list[k])
                if not os.path.exists(cur_file):
                    with open(cur_file, 'w') as temp_f:
                        pass
                    empty_cnt += 1

            if float(sys.version[:3]) >= 3.6:
                from tools.kitti_object_eval_python.evaluate import evaluate as kitti_evaluate 

                label_dir = os.path.join('./data/KITTI/object/testing', 'label_2')
                split_file = os.path.join('./data/KITTI', 'ImageSets', 'val.txt')
                final_output_dir = os.path.join("./result_dir", 'final_result', 'data')
                name_to_class = {'Car': 0, 'Pedestrian': 1, 'Cyclist': 2}
                ap_result_str, ap_dict = kitti_evaluate(
                    label_dir, final_output_dir, label_split_file=split_file,
                     current_class=name_to_class["Car"])

                logger.info("KITTI evaluate: {}, {}".format(ap_result_str, ap_dict))

            else:
                logger.info("kitti map only support python version >= 3.6")

    finally:
        eval_pyreader.reset()


if __name__ == "__main__":
    eval()
