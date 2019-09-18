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
from utils.config import cfg, load_config
from data import kitti_utils
from utils import calibration as calib

logging.root.handlers = []
FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)


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

# def save_kitti_result(sample_id, calib, bbox3d, kitti_output_dir, scores, img_shape):
def save_kitti_result(rets, seg_output_dir, kitti_output_dir, reader, classes):
    sample_id = rets['sample_id'][0]
    roi_scores_row = rets['roi_scores_row'][0]
    pts_rect = rets['pts_rect'][0]
    seg_mask = rets['seg_mask'][0]
    rpn_cls_label = rets['rpn_cls_label'][0]
    gt_boxes3d = rets['gt_boxes3d'][0]
    gt_boxes3d_num = rets['gt_boxes3d'][1]

    gt_box_idx = 0
    for i in range(len(sample_id)):
        s_id = sample_id[i, 0]

        seg_result_data = np.concatenate((pts_rect[i].reshape(-1, 3),
                                         rpn_cls_label[i].reshape(-1, 1),
                                         seg_mask[i].reshape(-1, 1)),
                                         axis=1).astype('float16')
        seg_output_file = os.path.join(seg_output_dir, '%06d.npy' % s_id)
        np.save(seg_output_file, seg_result_data)

        scores = roi_scores_row[i, :]
        bbox3d = gt_boxes3d[gt_box_idx: gt_box_idx + gt_boxes3d_num[0][i]]
        gt_box_idx += gt_boxes3d_num[0][i]
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
                # print('%s -1 -1 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f' %
                #       (cfg.CLASSES, alpha, img_boxes[k, 0], img_boxes[k, 1], img_boxes[k, 2], img_boxes[k, 3],
                #        bbox3d[k, 3], bbox3d[k, 4], bbox3d[k, 5], bbox3d[k, 0], bbox3d[k, 1], bbox3d[k, 2],
                #        bbox3d[k, 6], scores[k]), file=f)

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
        '--train_mode',
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
        default='checkpoints_adam_cosine/199',
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


# save_rpn_feature


def eval():
    args = parse_args()
    # check whether the installed paddle is compiled with GPU
    check_gpu(args.use_gpu)

    load_config(args.cfg)

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    if args.train_mode == 'rpn':
        cfg.RPN.ENABLED = True
        cfg.RCNN.ENABLED = False
    elif args.train_mode == 'rcnn':
        cfg.RCNN.ENABLED = True
        cfg.RPN.ENABLED = cfg.RPN.FIXED = True
    elif args.train_mode == 'rcnn_offline':
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
                    if args.train_mode == 'rpn' and args.save_rpn_feature else []
    eval_keys, eval_values = parse_outputs(eval_outputs, prog=eval_prog, extra_keys=extra_keys)

    eval_compile_prog = fluid.compiler.CompiledProgram(
            eval_prog).with_data_parallel()

    exe.run(startup)

    # load checkpoint
    assert os.path.isdir(args.ckpt_dir), "ckpt_dir {} not a directory".format(args.ckpt_dir)
    def if_exist(var):
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
    total_recalled_bbox_list, total_gt_bbox = [0] * 5, 0
    cnt = max_num = rpn_iou_avg = 0

    # get reader
    kitti_rcnn_reader = KittiRCNNReader(data_dir='./data',
                                    npoints=cfg.RPN.NUM_POINTS,
                                    split=cfg.TRAIN.SPLIT,
                                    mode='EVAL',
                                    classes=cfg.CLASSES,
                                    rcnn_eval_roi_dir=args.rcnn_eval_roi_dir,
                                    rcnn_eval_feature_dir=args.rcnn_eval_feature_dir)
    eval_reader = kitti_rcnn_reader.get_reader(args.batch_size, eval_feeds, False)
    eval_pyreader.decorate_sample_list_generator(eval_reader, place)

    eval_stat = Stat()
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
            eval_periods.append(period)

            # eval_stat.update(eval_keys, eval_outs)

            if args.save_rpn_feature:
                save_rpn_feature(rets, kitti_feature_dir)
                save_kitti_result(rets, seg_output_dir, kitti_output_dir, kitti_rcnn_reader, cfg.CLASSES)

            # if eval_iter % args.log_interval == 0:
            #     logger.info("3333333333333 {}".format(str(eval_keys)))
            #     log_str = ""
            #     for name, values in zip(eval_keys, eval_outs):
            #         log_str += "{}: {:.4f}, ".format(name, np.mean(values))
            #         logger.info("log {} {}".format(name, log_str))
            #     logger.info("[EVAL] Epoch {}, batch {}: {}time: {:.2f}".format(epoch_id, eval_iter, log_str, period))
            eval_iter += 1
    except fluid.core.EOFException:
        logger.info("[EVAL] Epoch {} finished, {}average time: {:.2f}".format(epoch_id, eval_stat.get_mean_log(), np.mean(eval_periods[2:])))
        save_model(exe, eval_prog, os.path.join(args.output_dir, str(epoch_id)))
        eval_stat.reset()
        eval_periods = []
    except Exception as e:
        logger.info(str(e))
    finally:
        eval_pyreader.reset()


if __name__ == "__main__":
    eval()
