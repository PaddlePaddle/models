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

import numpy as np
import datetime
import logging
import json
import pandas as pd
from models.bmn.bmn_utils import boundary_choose, soft_nms, bmn_post_processing
import time
import os
logger = logging.getLogger(__name__)


class MetricsCalculator():
    def __init__(self, cfg, name='BMN', mode='train'):
        self.name = name
        self.mode = mode  # 'train', 'valid', 'test', 'infer'
        self.tscale = cfg["MODEL"]["tscale"]
        self.dscale = cfg["MODEL"]["tscale"]
        self.subset = cfg[self.mode.upper()]["subset"]
        self.anno_file = cfg["MODEL"]["anno_file"]
        self.file_list = cfg["INFER"]["filelist"]
        self.get_dataset_dict()
        self.cols = ["xmin", "xmax", "score"]
        self.snippet_xmins = [1.0 / self.tscale * i for i in range(self.tscale)]
        self.snippet_xmaxs = [
            1.0 / self.tscale * i for i in range(1, self.tscale + 1)
        ]
        if self.mode == "test" or self.mode == "infer":
            self.output_path = cfg[self.mode.upper()]["output_path"]
            self.result_path = cfg[self.mode.upper()]["result_path"]
        self.reset()

    def get_dataset_dict(self):
        if self.mode == "infer":
            annos = json.load(open(self.file_list))
            self.video_dict = {}
            for video_name in annos.keys():
                self.video_dict[video_name] = annos[video_name]
        else:
            annos = json.load(open(self.anno_file))
            self.video_dict = {}
            for video_name in annos.keys():
                video_subset = annos[video_name]["subset"]
                if self.subset in video_subset:
                    self.video_dict[video_name] = annos[video_name]
        self.video_list = list(self.video_dict.keys())
        self.video_list.sort()

    def reset(self):
        logger.info('Resetting {} metrics...'.format(self.mode))
        self.aggr_loss = 0.0
        self.aggr_tem_loss = 0.0
        self.aggr_pem_reg_loss = 0.0
        self.aggr_pem_cls_loss = 0.0
        self.aggr_batch_size = 0
        if self.mode == 'test' or self.mode == "infer":
            if not os.path.exists(self.output_path):
                os.makedirs(self.output_path)

    def gen_props(self, pred_bm, pred_start, pred_end, fid):
        video_name = self.video_list[fid]
        pred_bm = pred_bm[0, 0, :, :] * pred_bm[0, 1, :, :]
        start_mask = boundary_choose(pred_start)
        start_mask[0] = 1.
        end_mask = boundary_choose(pred_end)
        end_mask[-1] = 1.
        score_vector_list = []
        for idx in range(self.dscale):
            for jdx in range(self.tscale):
                start_index = jdx
                end_index = start_index + idx
                if end_index < self.tscale and start_mask[
                        start_index] == 1 and end_mask[end_index] == 1:
                    xmin = self.snippet_xmins[start_index]
                    xmax = self.snippet_xmaxs[end_index]
                    xmin_score = pred_start[start_index]
                    xmax_score = pred_end[end_index]
                    bm_score = pred_bm[idx, jdx]
                    conf_score = xmin_score * xmax_score * bm_score
                    score_vector_list.append([xmin, xmax, conf_score])

        score_vector_list = np.stack(score_vector_list)
        video_df = pd.DataFrame(score_vector_list, columns=self.cols)
        video_df.to_csv(
            os.path.join(self.output_path, "%s.csv" % video_name), index=False)

    def accumulate(self, fetch_list):
        cur_batch_size = 1  # iteration counter,for test and inference, batch_size=1
        total_loss = fetch_list[0]
        tem_loss = fetch_list[1]
        pem_reg_loss = fetch_list[2]
        pem_cls_loss = fetch_list[3]

        self.aggr_loss += np.mean(np.array(total_loss))
        self.aggr_tem_loss += np.mean(np.array(tem_loss))
        self.aggr_pem_reg_loss += np.mean(np.array(pem_reg_loss))
        self.aggr_pem_cls_loss += np.mean(np.array(pem_cls_loss))
        self.aggr_batch_size += cur_batch_size

        if self.mode == 'test':
            pred_bm = np.array(fetch_list[4])
            pred_start = np.array(fetch_list[5])
            pred_end = np.array(fetch_list[6])
            fid = fetch_list[7][0][0]
            self.gen_props(pred_bm, pred_start, pred_end, fid)

    def accumulate_infer_results(self, fetch_list):
        pred_bm = np.array(fetch_list[0])
        pred_start = np.array(fetch_list[1][0])
        pred_end = np.array(fetch_list[2][0])
        fid = fetch_list[3][0]
        self.gen_props(pred_bm, pred_start, pred_end, fid)

    def finalize_metrics(self):
        self.avg_loss = self.aggr_loss / self.aggr_batch_size
        self.avg_tem_loss = self.aggr_tem_loss / self.aggr_batch_size
        self.avg_pem_reg_loss = self.aggr_pem_reg_loss / self.aggr_batch_size
        self.avg_pem_cls_loss = self.aggr_pem_cls_loss / self.aggr_batch_size
        if self.mode == 'test':
            bmn_post_processing(self.video_dict, self.subset, self.output_path,
                                self.result_path)

    def finalize_infer_metrics(self):
        bmn_post_processing(self.video_dict, self.subset, self.output_path,
                            self.result_path)

    def get_computed_metrics(self):
        json_stats = {}
        json_stats['avg_loss'] = self.avg_loss
        json_stats['avg_tem_loss'] = self.avg_tem_loss
        json_stats['avg_pem_reg_loss'] = self.avg_pem_reg_loss
        json_stats['avg_pem_cls_loss'] = self.avg_pem_cls_loss
        return json_stats
