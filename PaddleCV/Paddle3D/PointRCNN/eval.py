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
import logging
import multiprocessing
import numpy as np
from collections import OrderedDict 
import paddle
import paddle.fluid as fluid

from models.point_rcnn import PointRCNN
from data.kitti_rcnn_reader import KittiRCNNReader
from utils.run_utils import *
from utils.config import cfg, load_config, set_config_from_list
from utils.metric_utils import calc_iou_recall, rpn_metric, rcnn_metric

logging.root.handlers = []
FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)

np.random.seed(1024)  # use same seed
METRIC_PROC_NUM = 4


def parse_args():
    parser = argparse.ArgumentParser(
        "PointRCNN semantic segmentation train script")
    parser.add_argument(
        '--cfg',
        type=str,
        default='cfgs/default.yml',
        help='specify the config for training')
    parser.add_argument(
        '--eval_mode',
        type=str,
        default='rpn',
        required=True,
        help='specify the training mode')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help='evaluation batch size, default 1')
    parser.add_argument(
        '--weights',
        type=str,
        default='checkpoints/199',
        help='specify weights to be evaluated if needed')
    parser.add_argument(
        '--data_dir',
        type=str,
        default='./data',
        help='KITTI dataset root directory')
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
    parser.add_argument(
        '--set',
        dest='set_cfgs',
        default=None,
        nargs=argparse.REMAINDER,
        help='set extra config keys if needed.')
    args = parser.parse_args()
    return args


def eval():
    args = parse_args()
    print_arguments(args)
    # check whether the installed paddle is compiled with GPU
    # PointRCNN model can only run on GPU
    check_gpu(True)

    load_config(args.cfg)
    if args.set_cfgs is not None:
        set_config_from_list(args.set_cfgs)

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    if args.eval_mode == 'rpn':
        cfg.RPN.ENABLED = True
        cfg.RCNN.ENABLED = False
    elif args.eval_mode == 'rcnn':
        cfg.RCNN.ENABLED = True
        cfg.RPN.ENABLED = cfg.RPN.FIXED = True
        assert args.batch_size, "batch size must be 1 in rcnn evaluation"
    elif args.eval_mode == 'rcnn_offline':
        cfg.RCNN.ENABLED = True
        cfg.RPN.ENABLED = False
        assert args.batch_size, "batch size must be 1 in rcnn_offline evaluation"
    else:
        raise NotImplementedError("unkown eval mode: {}".format(args.eval_mode))

    place = fluid.CUDAPlace(0)
    exe = fluid.Executor(place)

    # build model
    startup = fluid.Program()
    eval_prog = fluid.Program()
    with fluid.program_guard(eval_prog, startup):
        with fluid.unique_name.guard():
            eval_model = PointRCNN(cfg, args.batch_size, True, 'TEST')
            eval_model.build()
            eval_loader = eval_model.get_loader()
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

    # load weights 
    if not os.path.isdir(args.weights):
        assert os.path.exists("{}.pdparams".format(args.weights)), \
                "Given resume weight {}.pdparams not exist.".format(args.weights)
    fluid.load(eval_prog, args.weights, exe)

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
    kitti_rcnn_reader = KittiRCNNReader(data_dir=args.data_dir,
                                        npoints=cfg.RPN.NUM_POINTS,
                                        split=cfg.TEST.SPLIT,
                                        mode='EVAL',
                                        classes=cfg.CLASSES,
                                        rcnn_eval_roi_dir=args.rcnn_eval_roi_dir,
                                        rcnn_eval_feature_dir=args.rcnn_eval_feature_dir)
    eval_reader = kitti_rcnn_reader.get_multiprocess_reader(args.batch_size, eval_feeds)
    eval_loader.set_sample_list_generator(eval_reader, place)

    thresh_list = [0.1, 0.3, 0.5, 0.7, 0.9]
    queue = multiprocessing.Queue(128)
    mgr = multiprocessing.Manager()
    lock = multiprocessing.Lock()
    mdict = mgr.dict()
    if cfg.RPN.ENABLED:
        mdict['exit_proc'] = 0
        mdict['total_gt_bbox'] = 0
        mdict['total_cnt'] = 0
        mdict['total_rpn_iou'] = 0
        for i in range(len(thresh_list)):
            mdict['total_recalled_bbox_list_{}'.format(i)] = 0

        p_list = []
        for i in range(METRIC_PROC_NUM):
            p_list.append(multiprocessing.Process(
                target=rpn_metric,
                args=(queue, mdict, lock, thresh_list, args.save_rpn_feature, kitti_feature_dir,
                      seg_output_dir, kitti_output_dir, kitti_rcnn_reader, cfg.CLASSES)))
            p_list[-1].start()
    
    if cfg.RCNN.ENABLED:
        for i in range(len(thresh_list)):
            mdict['total_recalled_bbox_list_{}'.format(i)] = 0
            mdict['total_roi_recalled_bbox_list_{}'.format(i)] = 0
        mdict['exit_proc'] = 0
        mdict['total_cls_acc'] = 0 
        mdict['total_cls_acc_refined'] = 0
        mdict['total_det_num'] = 0
        mdict['total_gt_bbox'] = 0
        p_list = []
        for i in range(METRIC_PROC_NUM):
            p_list.append(multiprocessing.Process(
                target=rcnn_metric,
                args=(queue, mdict, lock, thresh_list, kitti_rcnn_reader, roi_output_dir,
                      refine_output_dir, final_output_dir, args.save_result)
            ))
            p_list[-1].start()

    try:
        eval_loader.start()
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
        for i in range(METRIC_PROC_NUM):
            queue.put(None)
        while mdict['exit_proc'] < METRIC_PROC_NUM:
            time.sleep(1)
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
            for k in range(image_idx_list.__len__()):
                cur_file = os.path.join(final_output_dir, '%s.txt' % image_idx_list[k])
                if not os.path.exists(cur_file):
                    with open(cur_file, 'w') as temp_f:
                        pass

            if float(sys.version[:3]) >= 3.6:
                label_dir = os.path.join('./data/KITTI/object/training', 'label_2')
                split_file = os.path.join('./data/KITTI', 'ImageSets', 'val.txt')
                final_output_dir = os.path.join("./result_dir", 'final_result', 'data')
                name_to_class = {'Car': 0, 'Pedestrian': 1, 'Cyclist': 2}

                from tools.kitti_object_eval_python.evaluate import evaluate as kitti_evaluate 
                ap_result_str, ap_dict = kitti_evaluate(
                    label_dir, final_output_dir, label_split_file=split_file,
                     current_class=name_to_class["Car"])

                logger.info("KITTI evaluate: {}, {}".format(ap_result_str, ap_dict))

            else:
                logger.info("KITTI mAP only support python version >= 3.6, users can "
                            "run 'python3 tools/kitti_eval.py' to evaluate KITTI mAP.")

    finally:
        eval_loader.reset()


if __name__ == "__main__":
    eval()
