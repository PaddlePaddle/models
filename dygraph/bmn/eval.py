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

import paddle
import paddle.fluid as fluid
import numpy as np
import argparse
import pandas as pd
import os
import sys
import ast
import json
import logging

from reader import BMNReader
from model import BMN, bmn_loss_func
from bmn_utils import boundary_choose, bmn_post_processing
from config_utils import *

DATATYPE = 'float32'

logging.root.handlers = []
FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser("BMN test for performance evaluation.")
    parser.add_argument(
        '--config_file',
        type=str,
        default='bmn.yaml',
        help='path to config file of model')
    parser.add_argument(
        '--batch_size', type=int, default=1, help='eval batch size.')
    parser.add_argument(
        '--use_gpu',
        type=ast.literal_eval,
        default=True,
        help='default use gpu.')
    parser.add_argument(
        '--weights',
        type=str,
        default="checkpoint/bmn_paddle_dy_final",
        help='weight path, None to automatically download weights provided by Paddle.'
    )
    parser.add_argument(
        '--log_interval',
        type=int,
        default=1,
        help='mini-batch interval to log.')
    args = parser.parse_args()
    return args


def get_dataset_dict(cfg):
    anno_file = cfg.MODEL.anno_file
    annos = json.load(open(anno_file))
    subset = cfg.TEST.subset
    video_dict = {}
    for video_name in annos.keys():
        video_subset = annos[video_name]["subset"]
        if subset in video_subset:
            video_dict[video_name] = annos[video_name]
    video_list = list(video_dict.keys())
    video_list.sort()
    return video_dict, video_list


def gen_props(pred_bm, pred_start, pred_end, fid, video_list, cfg, mode='test'):
    if mode == 'infer':
        output_path = cfg.INFER.output_path
    else:
        output_path = cfg.TEST.output_path
    tscale = cfg.MODEL.tscale
    dscale = cfg.MODEL.dscale
    snippet_xmins = [1.0 / tscale * i for i in range(tscale)]
    snippet_xmaxs = [1.0 / tscale * i for i in range(1, tscale + 1)]
    cols = ["xmin", "xmax", "score"]

    video_name = video_list[fid]
    pred_bm = pred_bm[0, 0, :, :] * pred_bm[0, 1, :, :]
    start_mask = boundary_choose(pred_start)
    start_mask[0] = 1.
    end_mask = boundary_choose(pred_end)
    end_mask[-1] = 1.
    score_vector_list = []
    for idx in range(dscale):
        for jdx in range(tscale):
            start_index = jdx
            end_index = start_index + idx
            if end_index < tscale and start_mask[start_index] == 1 and end_mask[
                    end_index] == 1:
                xmin = snippet_xmins[start_index]
                xmax = snippet_xmaxs[end_index]
                xmin_score = pred_start[start_index]
                xmax_score = pred_end[end_index]
                bm_score = pred_bm[idx, jdx]
                conf_score = xmin_score * xmax_score * bm_score
                score_vector_list.append([xmin, xmax, conf_score])

    score_vector_list = np.stack(score_vector_list)
    video_df = pd.DataFrame(score_vector_list, columns=cols)
    video_df.to_csv(
        os.path.join(output_path, "%s.csv" % video_name), index=False)


# Performance Evaluation
def test_bmn(args):
    config = parse_config(args.config_file)
    test_config = merge_configs(config, 'test', vars(args))
    print_configs(test_config, "Test")

    if not os.path.isdir(test_config.TEST.output_path):
        os.makedirs(test_config.TEST.output_path)
    if not os.path.isdir(test_config.TEST.result_path):
        os.makedirs(test_config.TEST.result_path)

    if not args.use_gpu:
        place = fluid.CPUPlace()
    else:
        place = fluid.CUDAPlace(0)

    with fluid.dygraph.guard(place):
        bmn = BMN(test_config)

        # load checkpoint
        if args.weights:
            assert os.path.exists(args.weights + '.pdparams'
                                  ), "Given weight dir {} not exist.".format(
                                      args.weights)

        logger.info('load test weights from {}'.format(args.weights))
        model_dict, _ = fluid.load_dygraph(args.weights)
        bmn.set_dict(model_dict)

        reader = BMNReader(mode="test", cfg=test_config)
        test_reader = reader.create_reader()

        aggr_loss = 0.0
        aggr_tem_loss = 0.0
        aggr_pem_reg_loss = 0.0
        aggr_pem_cls_loss = 0.0
        aggr_batch_size = 0
        video_dict, video_list = get_dataset_dict(test_config)

        bmn.eval()
        for batch_id, data in enumerate(test_reader()):
            video_feat = np.array([item[0] for item in data]).astype(DATATYPE)
            gt_iou_map = np.array([item[1] for item in data]).astype(DATATYPE)
            gt_start = np.array([item[2] for item in data]).astype(DATATYPE)
            gt_end = np.array([item[3] for item in data]).astype(DATATYPE)
            video_idx = [item[4] for item in data][0]  #batch_size=1 by default

            x_data = fluid.dygraph.base.to_variable(video_feat)
            gt_iou_map = fluid.dygraph.base.to_variable(gt_iou_map)
            gt_start = fluid.dygraph.base.to_variable(gt_start)
            gt_end = fluid.dygraph.base.to_variable(gt_end)
            gt_iou_map.stop_gradient = True
            gt_start.stop_gradient = True
            gt_end.stop_gradient = True

            pred_bm, pred_start, pred_end = bmn(x_data)
            loss, tem_loss, pem_reg_loss, pem_cls_loss = bmn_loss_func(
                pred_bm, pred_start, pred_end, gt_iou_map, gt_start, gt_end,
                test_config)

            pred_bm = pred_bm.numpy()
            pred_start = pred_start[0].numpy()
            pred_end = pred_end[0].numpy()
            aggr_loss += np.mean(loss.numpy())
            aggr_tem_loss += np.mean(tem_loss.numpy())
            aggr_pem_reg_loss += np.mean(pem_reg_loss.numpy())
            aggr_pem_cls_loss += np.mean(pem_cls_loss.numpy())
            aggr_batch_size += 1

            if batch_id % args.log_interval == 0:
                logger.info("Processing................ batch {}".format(
                    batch_id))

            gen_props(
                pred_bm,
                pred_start,
                pred_end,
                video_idx,
                video_list,
                test_config,
                mode='test')

        avg_loss = aggr_loss / aggr_batch_size
        avg_tem_loss = aggr_tem_loss / aggr_batch_size
        avg_pem_reg_loss = aggr_pem_reg_loss / aggr_batch_size
        avg_pem_cls_loss = aggr_pem_cls_loss / aggr_batch_size

        logger.info('[EVAL] \tAvg_oss = {}, \tAvg_tem_loss = {}, \tAvg_pem_reg_loss = {}, \tAvg_pem_cls_loss = {}'.format(
            '%.04f' % avg_loss, '%.04f' % avg_tem_loss, \
            '%.04f' % avg_pem_reg_loss, '%.04f' % avg_pem_cls_loss))

        logger.info("Post_processing....This may take a while")
        bmn_post_processing(video_dict, test_config.TEST.subset,
                            test_config.TEST.output_path,
                            test_config.TEST.result_path)
        logger.info("[EVAL] eval finished")


if __name__ == '__main__':
    args = parse_args()
    test_bmn(args)
