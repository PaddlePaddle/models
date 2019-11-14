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
from models.bsn.bsn_utils import pgm_gen_proposal, pgm_gen_feature
import time
logger = logging.getLogger(__name__)
import os


class MetricsCalculator():
    def __init__(self, cfg, name='BsnTem', mode='train'):
        self.name = name
        self.mode = mode  # 'train', 'valid', 'test', 'infer'
        self.tscale = cfg["MODEL"]["tscale"]
        self.subset = cfg[self.mode.upper()][
            "subset"]  # 'train', 'validation', 'train_val'
        self.anno_file = cfg["MODEL"]["anno_file"]
        self.file_list = cfg["INFER"]["filelist"]
        self.get_pgm_cfg(cfg)
        self.get_dataset_dict()
        self.cols = ["xmin", "xmax", "score"]
        self.snippet_xmins = [1.0 / self.tscale * i for i in range(self.tscale)]
        self.snippet_xmaxs = [
            1.0 / self.tscale * i for i in range(1, self.tscale + 1)
        ]
        if self.mode == "test" or self.mode == "infer":
            self.output_path_tem = cfg[self.mode.upper()]["output_path_tem"]
            self.output_path_pgm_feature = cfg[self.mode.upper()][
                "output_path_pgm_feature"]
            self.output_path_pgm_proposal = cfg[self.mode.upper()][
                "output_path_pgm_proposal"]
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
                if self.subset == "train_val":
                    if "train" in video_subset or "validation" in video_subset:
                        self.video_dict[video_name] = annos[video_name]
                else:
                    if self.subset in video_subset:
                        self.video_dict[video_name] = annos[video_name]
        self.video_list = list(self.video_dict.keys())
        self.video_list.sort()

    def get_pgm_cfg(self, cfg):
        self.pgm_config = {}
        if self.mode == "test" or self.mode == "infer":
            self.pgm_config["tscale"] = self.tscale
            self.pgm_config["pgm_threshold"] = cfg["MODEL"]["pgm_threshold"]
            self.pgm_config["pgm_top_K_train"] = cfg["MODEL"]["pgm_top_K_train"]
            self.pgm_config["pgm_top_K"] = cfg["MODEL"]["pgm_top_K"]
            self.pgm_config["bsp_boundary_ratio"] = cfg["MODEL"][
                "bsp_boundary_ratio"]
            self.pgm_config["num_sample_start"] = cfg["MODEL"][
                "num_sample_start"]
            self.pgm_config["num_sample_end"] = cfg["MODEL"]["num_sample_end"]
            self.pgm_config["num_sample_action"] = cfg["MODEL"][
                "num_sample_action"]
            self.pgm_config["num_sample_perbin"] = cfg["MODEL"][
                "num_sample_perbin"]
            self.pgm_config["pgm_thread"] = cfg["MODEL"]["pgm_thread"]

    def reset(self):
        logger.info('Resetting {} metrics...'.format(self.mode))
        self.aggr_loss = 0.0
        self.aggr_start_loss = 0.0
        self.aggr_end_loss = 0.0
        self.aggr_action_loss = 0.0
        self.aggr_batch_size = 0
        if self.mode == 'test' or self.mode == 'infer':
            if not os.path.exists(self.output_path_tem):
                os.makedirs(self.output_path_tem)
            if not os.path.exists(self.output_path_pgm_feature):
                os.makedirs(self.output_path_pgm_feature)
            if not os.path.exists(self.output_path_pgm_proposal):
                os.makedirs(self.output_path_pgm_proposal)

    def save_results(self, pred_tem, fid):
        batch_size = pred_tem.shape[0]
        for i in range(batch_size):
            if self.mode == 'test':
                video_name = self.video_list[fid[i][0]]
            elif self.mode == 'infer':
                video_name = self.video_list[fid[i]]
            pred_start = pred_tem[i, 0, :]
            pred_end = pred_tem[i, 1, :]
            pred_action = pred_tem[i, 2, :]
            output_tem = np.stack([pred_start, pred_end, pred_action], axis=1)
            video_df = pd.DataFrame(
                output_tem, columns=["start", "end", "action"])
            video_df.to_csv(
                os.path.join(self.output_path_tem, video_name + ".csv"),
                index=False)

    def accumulate(self, fetch_list):
        cur_batch_size = 1  # iteration counter
        total_loss = fetch_list[0]
        start_loss = fetch_list[1]
        end_loss = fetch_list[2]
        action_loss = fetch_list[3]

        self.aggr_loss += np.mean(np.array(total_loss))
        self.aggr_start_loss += np.mean(np.array(start_loss))
        self.aggr_end_loss += np.mean(np.array(end_loss))
        self.aggr_action_loss += np.mean(np.array(action_loss))
        self.aggr_batch_size += cur_batch_size

        if self.mode == 'test':
            pred_tem = np.array(fetch_list[4])
            fid = fetch_list[5]
            self.save_results(pred_tem, fid)

    def accumulate_infer_results(self, fetch_list):
        pred_tem = np.array(fetch_list[0])
        fid = fetch_list[1]
        self.save_results(pred_tem, fid)

    def finalize_metrics(self):
        self.avg_loss = self.aggr_loss / self.aggr_batch_size
        self.avg_start_loss = self.aggr_start_loss / self.aggr_batch_size
        self.avg_end_loss = self.aggr_end_loss / self.aggr_batch_size
        self.avg_action_loss = self.aggr_action_loss / self.aggr_batch_size
        if self.mode == 'test':
            print("start generate proposals of %s subset" % self.subset)
            pgm_gen_proposal(self.video_dict, self.pgm_config,
                             self.output_path_tem,
                             self.output_path_pgm_proposal)
            print("finish generate proposals of %s subset" % self.subset)
            print("start generate proposals feature of %s subset" % self.subset)
            pgm_gen_feature(self.video_dict, self.pgm_config,
                            self.output_path_tem, self.output_path_pgm_proposal,
                            self.output_path_pgm_feature)
            print("finish generate proposals feature of %s subset" %
                  self.subset)

    def finalize_infer_metrics(self):
        print("start generate proposals of %s subset" % self.subset)
        pgm_gen_proposal(self.video_dict, self.pgm_config, self.output_path_tem,
                         self.output_path_pgm_proposal)
        print("finish generate proposals of %s subset" % self.subset)
        print("start generate proposals feature of %s subset" % self.subset)
        pgm_gen_feature(self.video_dict, self.pgm_config, self.output_path_tem,
                        self.output_path_pgm_proposal,
                        self.output_path_pgm_feature)
        print("finish generate proposals feature of %s subset" % self.subset)

    def get_computed_metrics(self):
        json_stats = {}
        json_stats['avg_loss'] = self.avg_loss
        json_stats['avg_start_loss'] = self.avg_start_loss
        json_stats['avg_end_loss'] = self.avg_end_loss
        json_stats['avg_action_loss'] = self.avg_action_loss
        return json_stats
