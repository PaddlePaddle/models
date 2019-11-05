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

import os
import sys
import time
import shutil
import argparse
import ast
import logging
import numpy as np
import paddle
import paddle.fluid as fluid
from paddle.fluid.layers import control_flow
from paddle.fluid.contrib.extend_optimizer import extend_with_decoupled_weight_decay
import paddle.fluid.layers.learning_rate_scheduler as lr_scheduler

from models.point_rcnn import PointRCNN
from data.kitti_rcnn_reader import KittiRCNNReader
from utils.run_utils import check_gpu, parse_outputs, Stat 
from utils.box_utils import boxes_iou3d
from utils.config import cfg, load_config
from data import kitti_utils
from utils import calibration as calib
import utils.cyops.kitti_utils as kitti_utils 
from utils.cyops.kitti_utils import rotate_pc_along_y_np
#from utils.cyops.iou3d_utils import boxes_iou3d
#from utils.iou3d_utils import boxes_iou3d 
from utils.box_utils import boxes_iou3d 

logging.root.handlers = []
FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)

np.random.seed(1024) # use same random seed

def parse_args():
    parser = argparse.ArgumentParser("PointRCNN semantic segmentation train script")
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
        # default='checkpoints_adamw_cosine/199',
        default='checkpoints/198',
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
        '--rcnn_eval_roi_dir',
        type=str,
        default=None,
	help='specify the saved rois for rcnn evaluation when using rcnn_offline mode')
    parser.add_argument(
        '--rcnn_eval_feature_dir',
        type=str,
        default=None,
	help='specify the saved features for rcnn evaluation when using rcnn_offline mode')
    parser.add_argument(
        '--log_interval',
        type=int,
        default=1,
        help='mini-batch interval to log.')
    args = parser.parse_args()
    return args


# def save_rpn_feature(seg_result, rpn_scores_raw, pts_features, backbone_xyz, backbone_features, kitti_features_dir,
#                       sample_id):
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
        intensity_file = os.path.join(kitti_features_dir, '%06d_intensity.npy' % s_id)
        np.save(output_file, backbone_feature[i])
        np.save(xyz_file, backbone_xyz[i])
        np.save(seg_file, seg_mask[i])
        np.save(intensity_file, pts_intensity)
        rpn_scores_raw_file = os.path.join(kitti_features_dir, '%06d_rawscore.npy' % s_id)
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
        box_valid_mask = np.logical_and(img_boxes_w < img_shape[1] * 0.8, img_boxes_h < img_shape[0] * 0.8)
        print("img_boxes shape:", img_boxes.shape, np.sum(box_valid_mask))

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
        cur_gt_boxes3d = gt_boxes3d[gt_box_idx: gt_box_idx + gt_boxes3d_num[0][i]]
        gt_box_idx += gt_boxes3d_num[0][i]

	k = cur_gt_boxes3d.__len__() - 1
	while k >= 0 and np.sum(cur_gt_boxes3d[k]) == 0:
	    k -= 1
	cur_gt_boxes3d = cur_gt_boxes3d[:k + 1]

	if cur_gt_boxes3d.shape[0] > 0:
	    iou3d = boxes_iou3d(cur_boxes3d, cur_gt_boxes3d[:, 0:7])
            # logger.info("iou3d {}".format(iou3d))
	    gt_max_iou = iou3d.max(axis=0)

	    for idx, thresh in enumerate(thresh_list):
		recalled_bbox_list[idx] += np.sum(gt_max_iou > thresh)
	    gt_box_num += cur_gt_boxes3d.__len__()

	fg_mask = cur_rpn_cls_label > 0
	correct = np.sum(np.logical_and(cur_seg_mask == cur_rpn_cls_label, fg_mask))
	union = np.sum(fg_mask) + np.sum(cur_seg_mask > 0) - correct
	rpn_iou = float(correct) / max(float(union), 1.0)
	rpn_iou_sum += rpn_iou
        logger.info('sample_id {} rpn_iou {} correct {} union {} fg_mask {}'.format(sample_id, rpn_iou, correct, union, np.sum(fg_mask)))

    return len(gt_boxes3d_num), gt_box_num, rpn_iou_sum, recalled_bbox_list

def rotate_pc_along_y_py(pc, rot_angle):
    
    cosa = np.cos(rot_angle).reshape(-1, 1)
    sina = np.sin(rot_angle).reshape(-1, 1)
    raw_1 = np.concatenate((cosa, -sina), axis=1)
    raw_2 = np.concatenate((sina, cosa), axis=1)
    # # (N, 2, 2)
    R = np.concatenate((np.expand_dims(raw_1, axis=1), np.expand_dims(raw_2, axis=1)), axis=1)
    
    pc_temp = pc[:, [0, 2]]
    pc_temp = np.expand_dims(pc_temp, axis=1)
    pc[:, [0, 2]] = np.squeeze(np.matmul(pc_temp, R.transpose(0, 2, 1)), axis=1)
    return pc

def decode_bbox_target(roi_box3d, pred_reg, loc_scope, loc_bin_size, num_head_bin, anchor_size,
                       get_xz_fine=True, get_y_by_bin=False, loc_y_scope=0.5, loc_y_bin_size=0.25, get_ry_fine=False):
    roi_box3d = roi_box3d.reshape((-1, roi_box3d.shape[-1]))
    per_loc_bin_num = int(loc_scope / loc_bin_size) * 2
    loc_y_bin_num = int(loc_y_scope / loc_y_bin_size) * 2

    # recover xz localization
    x_bin_l = 0 
    x_bin_r = per_loc_bin_num 
    z_bin_l, z_bin_r = per_loc_bin_num, per_loc_bin_num * 2
    start_offset = z_bin_r
    x_bin = np.argmax(pred_reg[:, x_bin_l: x_bin_r], axis=1)
    z_bin = np.argmax(pred_reg[:, z_bin_l: z_bin_r], axis=1)
    pos_x = x_bin * loc_bin_size + loc_bin_size / 2 - loc_scope
    pos_z = z_bin * loc_bin_size + loc_bin_size / 2 - loc_scope
    if get_xz_fine:
        x_res_l, x_res_r = per_loc_bin_num * 2, per_loc_bin_num * 3
        z_res_l, z_res_r = per_loc_bin_num * 3, per_loc_bin_num * 4
        start_offset = z_res_r

        x_res_norm = np.take(pred_reg[:, x_res_l: x_res_r], indices=np.expand_dims(x_bin, axis=1))
        x_res_norm = np.squeeze(x_res_norm, axis=1)
        z_res_norm = np.take(pred_reg[:, z_res_l: z_res_r], indices=np.expand_dims(z_bin, axis=1))
        z_res_norm = np.squeeze(z_res_norm, axis=1)
        x_res = x_res_norm * loc_bin_size
        z_res = z_res_norm * loc_bin_size

        pos_x += x_res
        pos_z += z_res
    # recover y localization
    if get_y_by_bin:
        y_bin_l = start_offset
        y_bin_r = start_offset + loc_y_bin_num
        y_res_l = y_bin_r  
        y_res_r = y_bin_r + loc_y_bin_num
        start_offset = y_res_r
        y_bin = np.argmax(pred_reg[:, y_bin_l: y_bin_r], axis=1)
        y_res_norm = np.take(pred_reg[:, y_res_l: y_res_r], indices=np.expand_dims(y_bin, axis=1))
        y_res_norm = np.squeeze(y_res_normk, axis=1)
        
        y_res = y_res_norm * loc_y_bin_size
        pos_y = y_bin.float() * loc_y_bin_size + loc_y_bin_size / 2 - loc_y_scope + y_res
        pos_y = pos_y + roi_box3d[:, 1]
    else:
        y_offset_l = start_offset
        y_offset_r = start_offset + 1
        start_offset = y_offset_r
        pos_y = roi_box3d[:, 1] + pred_reg[:, y_offset_l]

    # recover ry rotation
    ry_bin_l, ry_bin_r = start_offset, start_offset + num_head_bin
    ry_res_l, ry_res_r = ry_bin_r, ry_bin_r + num_head_bin

    ry_bin = np.argmax(pred_reg[:, ry_bin_l: ry_bin_r], axis=1)
    ry_res_norm = np.take(pred_reg[:, ry_res_l: ry_res_r], indices=np.expand_dims(ry_bin, axis=1))
    ry_res_norm = np.squeeze(ry_res_norm, axis=1)
    if get_ry_fine:
        # divide pi/2 into several bins
        angle_per_class = (np.pi / 2) / num_head_bin
        ry_res = ry_res_norm * (angle_per_class / 2)
        ry = (ry_bin * angle_per_class + angle_per_class / 2) + ry_res - np.pi / 4
    else:
        angle_per_class = (2 * np.pi) / num_head_bin
        ry_res = ry_res_norm * (angle_per_class / 2)
        
        # bin_center is (0, 30, 60, 90, 120, ..., 270, 300, 330)
        ry = (ry_bin.float() * angle_per_class + ry_res) % (2 * np.pi)
        ry[ry > np.pi] -= 2 * np.pi
    # recover size
    size_res_l, size_res_r = ry_res_r, ry_res_r + 3
    assert size_res_r == pred_reg.shape[1]

    size_res_norm = pred_reg[:, size_res_l: size_res_r]
    hwl = size_res_norm * anchor_size + anchor_size
    # shift to original coords
    roi_center = roi_box3d[:, 0:3]
    shift_ret_box3d = np.concatenate(
            (pos_x.reshape(-1, 1), pos_y.reshape(-1, 1), pos_z.reshape(-1, 1), hwl, ry.reshape(-1, 1)), axis=1)
    
    ret_box3d = shift_ret_box3d
    if roi_box3d.shape[1] == 7:
        roi_ry = roi_box3d[:, 6]*(-1)
        ret_box3d = rotate_pc_along_y_py(shift_ret_box3d, roi_ry)
        ret_box3d[:, 6] += roi_ry 
    ret_box3d[:, [0, 2]] += roi_center[:, [0, 2]]
    return ret_box3d

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

    extra_keys = ['sample_id', 'pts_rect', 'pts_features', 'pts_input', 'gt_boxes3d', 'rpn_cls_label'] \
                    if args.eval_mode == 'rpn' and args.save_rpn_feature else ['sample_id', 'rpn_cls_label', 'gt_boxes3d']
    eval_keys, eval_values = parse_outputs(eval_outputs, prog=eval_prog, extra_keys=extra_keys)

    eval_compile_prog = fluid.compiler.CompiledProgram(
            eval_prog).with_data_parallel()

    exe.run(startup)

    # load checkpoint
    assert os.path.isdir(args.ckpt_dir), "ckpt_dir {} not a directory".format(args.ckpt_dir)
    def if_exist(var):
        if os.path.exists(os.path.join(args.ckpt_dir, var.name)):
            logger.info("Load {}".format(var.name))
        return os.path.exists(os.path.join(args.ckpt_dir, var.name))
    fluid.io.load_vars(exe, args.ckpt_dir, eval_prog, predicate=if_exist)

    if args.save_rpn_feature:
        kitti_feature_dir = os.path.join(args.output_dir, 'features')
        if os.path.exists(kitti_feature_dir):
            shutil.rmtree(kitti_feature_dir)
        os.makedirs(kitti_feature_dir)
        kitti_output_dir = os.path.join(args.output_dir, 'detections', 'data')
        if os.path.exists(kitti_output_dir):
            shutil.rmtree(kitti_output_dir)
        os.makedirs(kitti_output_dir)
        seg_output_dir = os.path.join(args.output_dir, 'seg_result')
        if os.path.exists(seg_output_dir):
            shutil.rmtree(seg_output_dir)
        os.makedirs(seg_output_dir)

    thresh_list = [0.1, 0.3, 0.5, 0.7, 0.9]
    total_recalled_bbox_list = [0] * 5
    total_gt_bbox = 0
    total_cnt = 0
    total_rpn_iou = 0.

    # get reader
    kitti_rcnn_reader = KittiRCNNReader(data_dir='./data',
                                    npoints=cfg.RPN.NUM_POINTS,
                                    split=cfg.TEST.SPLIT,
                                    mode='EVAL',
                                    classes=cfg.CLASSES,
                                    rcnn_eval_roi_dir=args.rcnn_eval_roi_dir,
                                    rcnn_eval_feature_dir=args.rcnn_eval_feature_dir)
    eval_reader = kitti_rcnn_reader.get_reader(args.batch_size, eval_feeds, False)
    eval_pyreader.decorate_sample_list_generator(eval_reader, place)

    try:
        eval_pyreader.start()
        eval_iter = 0
        eval_periods = []
        while True:
            cur_time = time.time()
            eval_outs = exe.run(eval_compile_prog, fetch_list=eval_values, return_numpy=False)
            rets = {k: (np.array(v), v.recursive_sequence_lengths()) 
                        for k, v in zip(eval_keys, eval_outs)}
            period = time.time() - cur_time
            logger.info("exe run: {}s".format(period))
            eval_periods.append(period)

            if cfg.RPN.ENABLED:
                if not args.save_rpn_feature:
                    cnt, gt_box_num, rpn_iou_sum, recalled_bbox_list = calc_iou_recall(rets, thresh_list)
                    total_cnt += cnt
                    total_gt_bbox += gt_box_num
                    total_rpn_iou += rpn_iou_sum
                    total_recalled_bbox_list = [x + y for x, y in zip(total_recalled_bbox_list, recalled_bbox_list)]
                    logger.info("total_cnt={}, total_gt_bbox={}, total_rpn_iou={}, total_recalled_bbox_list={}".format(total_cnt, total_gt_bbox, total_rpn_iou, total_recalled_bbox_list))
                else:
                    save_rpn_feature(rets, kitti_feature_dir)
                    save_kitti_result(rets, seg_output_dir, kitti_output_dir, kitti_rcnn_reader, cfg.CLASSES)
            elif cfg.RCNN.ENABLED:
                cnt = 0
                final_total = 0  
                total_cls_acc = 0 
                total_cls_acc_refined = 0
                total_roi_recalled_bbox_list = [0] * 5
                rcnn_cls = rets_dict['rcnn_cls']
                rcnn_reg = rets_dict['rcnn_reg']
                roi_boxes3d = rets_dict['roi_boxes3d']
                # roi_scores = rets['roi_scores']
                # bounding box regression
                anchor_size = cfg.CLS_MEAN_SIZE[0]
                #if cfg.RCNN.SIZE_RES_ON_ROI:
                #    roi_size = rets_dict['roi_size']
                #    anchor_size = roi_size 
                #print("roi_boxes3d, rcnn_reg: ", roi_boxes3d.shape, rcnn_reg.shape)
                pred_boxes3d = decode_bbox_target(
                    roi_boxes3d, 
                    rcnn_reg,
                    anchor_size=anchor_size,
                    loc_scope=1.5, #cfg.RCNN.LOC_SCOPE,
                    loc_bin_size=0.5, #cfg.RCNN.LOC_BIN_SIZE,
                    num_head_bin=9, #cfg.RCNN.NUM_HEAD_BIN,
                    get_xz_fine=True, 
                    get_y_by_bin=False, #cfg.RCNN.LOC_Y_BY_BIN,
                    loc_y_scope=0.5, #cfg.RCNN.LOC_Y_SCOPE, 
                    loc_y_bin_size=0.25, #cfg.RCNN.LOC_Y_BIN_SIZE,
                    get_ry_fine=True
                )
                #print("pred_boxes3d: ", pred_boxes3d.shape)

                # scoring
                if rcnn_cls.shape[1] == 1:
                    norm_scores = rets_dict['norm_scores']
                    pred_classes = norm_scores > cfg.RCNN.SCORE_THRESH
                else:
                    pred_classes = np.argmax(rcnn_cls, axis=1).reshape(-1)

                # evaluation
                disp_dict = {'run time': period}
                if True: # eval mode 
                    gt_iou = rets_dict['gt_iou']
                    #print("gt_iou: ", gt_iou.shape)
                    # (-1, -1, 7)
                    gt_boxes3d = rets_dict['gt_boxes3d']
                    
                    # recall
                    gt_num = gt_boxes3d.shape[0]
                    if gt_num > 0:
                        gt_boxes3d = gt_boxes3d.reshape((-1,7))
                        iou3d = boxes_iou3d(pred_boxes3d, gt_boxes3d)
                        gt_max_iou = iou3d.max(axis=0)
                        refined_iou = iou3d.max(axis=1)

                        for idx, thresh in enumerate(thresh_list):
                            total_recalled_bbox_list[idx] += (gt_max_iou > thresh).sum().item()
                        recalled_num = (gt_max_iou > 0.7).sum().item()
                        total_gt_bbox += gt_num
                        roi_boxes3d = roi_boxes3d.reshape((-1,7))
                        iou3d_in = boxes_iou3d(roi_boxes3d, gt_boxes3d)
                        gt_max_iou_in = iou3d_in.max(axis=0)

                        for idx, thresh in enumerate(thresh_list):
                            total_roi_recalled_bbox_list[idx] += (gt_max_iou_in > thresh).sum().item()
                    # classification accuracy
                    cls_label = gt_iou > cfg.RCNN.CLS_FG_THRESH
                    cls_valid_mask = (gt_iou >= cfg.RCNN.CLS_FG_THRESH) | (gt_iou <= cfg.RCNN.CLS_BG_THRESH)
                    cls_acc = np.sum(pred_classes == cls_label * cls_valid_mask) / max(np.sum(cls_valid_mask), 1.0)
                    iou_thresh = 0.7 if cfg.CLASSES == 'Car' else 0.5
                    cls_label_refined = (gt_iou >= iou_thresh)
                    cls_acc_refined = (pred_classes == cls_label_refined).sum() / max(cls_label_refined.shape[0], 1.0)
                    total_cls_acc += cls_acc.item()
                    total_cls_acc_refined += cls_acc_refined.item()
                    disp_dict['recall'] = '%d/%d' % (total_recalled_bbox_list[3], total_gt_bbox)
                    disp_dict['cls_acc_refined'] = '%.2f' % cls_acc_refined.item()
                    print("disp_dict: ", disp_dict)
                
                # image_shape = dataset.get_image_shape(sample_id)
                # if args.save_result:
                #     # save roi and refine results
                #     roi_boxes3d_np = roi_boxes3d.cpu().numpy()
                #     pred_boxes3d_np = pred_boxes3d.cpu().numpy()
                #     calib = dataset.get_calib(sample_id)

                #     save_kitti_format(sample_id, calib, roi_boxes3d_np, roi_output_dir, roi_scores, image_shape)
                #     save_kitti_format(sample_id, calib, pred_boxes3d_np, refine_output_dir, raw_scores.cpu().numpy(),
                #                     image_shape)

                # NMS and scoring
                # scores thresh
                # inds = norm_scores > cfg.RCNN.SCORE_THRESH
                # if inds.sum() == 0:
                #     continue

                # pred_boxes3d_selected = pred_boxes3d[inds]
                # raw_scores_selected = raw_scores[inds]

                # # NMS thresh
                # boxes_bev_selected = kitti_utils.boxes3d_to_bev_torch(pred_boxes3d_selected)
                # keep_idx = iou3d_utils.nms_gpu(boxes_bev_selected, raw_scores_selected, cfg.RCNN.NMS_THRESH)
                # pred_boxes3d_selected = pred_boxes3d_selected[keep_idx]

                # scores_selected = raw_scores_selected[keep_idx]
                # pred_boxes3d_selected, scores_selected = pred_boxes3d_selected.cpu().numpy(), scores_selected.cpu().numpy()

                # calib = dataset.get_calib(sample_id)
                # final_total += pred_boxes3d_selected.shape[0]
                # save_kitti_format(sample_id, calib, pred_boxes3d_selected, final_output_dir, scores_selected, image_shape)               
            else:
                print("Only support RPN/RCNN!!!")


            logger.info("[EVAL] eval iter {}".format(eval_iter))

            eval_iter += 1
    except fluid.core.EOFException:
        logger.info("[EVAL] total {} iter finished, average time: {:.2f}".format(eval_iter, np.mean(eval_periods[2:])))
        if not args.save_rpn_feature:
            avg_rpn_iou = total_rpn_iou / max(len(kitti_rcnn_reader), 1.)
            logger.info("average rpn iou: {:.3f}".format(avg_rpn_iou))
            for idx, thresh in enumerate(thresh_list):
                recall = float(total_recalled_bbox_list[idx]) / max(total_gt_bbox, 1.)
                logger.info("total bbox recall(thresh={:.3f}): {} / {} = {:.3f}".format(
                    thresh, total_recalled_bbox_list[idx], total_gt_bbox, recall))
    finally:
        eval_pyreader.reset()


if __name__ == "__main__":
    eval()
