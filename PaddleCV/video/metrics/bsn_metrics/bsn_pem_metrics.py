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
from models.bsn.bsn_utils import soft_nms, bsn_post_processing
import time
logger = logging.getLogger(__name__)
import os


class MetricsCalculator():
    def __init__(self, cfg, name='BsnPem', mode='train'):
        self.name = name
        self.mode = mode  # 'train', 'valid', 'test', 'infer'
        self.subset = cfg[self.mode.upper()][
            "subset"]  # 'train', 'validation', 'test'
        self.anno_file = cfg["MODEL"]["anno_file"]
        self.file_list = cfg["INFER"]["filelist"]
        self.get_dataset_dict()
        if self.mode == "test" or self.mode == "infer":
            self.output_path_pem = cfg[self.mode.upper()]["output_path_pem"]
            self.result_path_pem = cfg[self.mode.upper()]["result_path_pem"]
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
        self.aggr_batch_size = 0
        if self.mode == 'test' or self.mode == 'infer':
            if not os.path.exists(self.output_path_pem):
                os.makedirs(self.output_path_pem)

    def save_results(self, pred_iou, props_info, fid):
        if self.mode == 'infer':
            video_name = self.video_list[fid[0]]
        else:
            video_name = self.video_list[fid[0][0]]
        df = pd.DataFrame()
        df["xmin"] = props_info[0, :, 0]
        df["xmax"] = props_info[0, :, 1]
        df["xmin_score"] = props_info[0, :, 2]
        df["xmax_score"] = props_info[0, :, 3]
        df["iou_score"] = pred_iou.squeeze()
        df.to_csv(
            os.path.join(self.output_path_pem, video_name + ".csv"),
            index=False)

    def accumulate(self, fetch_list):
        cur_batch_size = 1  # iteration counter
        total_loss = fetch_list[0]

        self.aggr_loss += np.mean(np.array(total_loss))
        self.aggr_batch_size += cur_batch_size

        if self.mode == 'test':
            pred_iou = np.array(fetch_list[1])
            props_info = np.array(fetch_list[2])
            fid = np.array(fetch_list[3])
            self.save_results(pred_iou, props_info, fid)

    def accumulate_infer_results(self, fetch_list):
        pred_iou = np.array(fetch_list[0])
        props_info = np.array([item[0] for item in fetch_list[1]])
        fid = [item[1] for item in fetch_list[1]]
        self.save_results(pred_iou, props_info, fid)

    def finalize_metrics(self):
        self.avg_loss = self.aggr_loss / self.aggr_batch_size
        if self.mode == 'test':
            bsn_post_processing(self.video_dict, self.subset,
                                self.output_path_pem, self.result_path_pem)

    def finalize_infer_metrics(self):
        bsn_post_processing(self.video_dict, self.subset, self.output_path_pem,
                            self.result_path_pem)

    def get_computed_metrics(self):
        json_stats = {}
        json_stats['avg_loss'] = self.avg_loss
        return json_stats
