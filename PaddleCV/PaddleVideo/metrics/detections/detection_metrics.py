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

from models.ctcn.ctcn_utils import BoxCoder

logger = logging.getLogger(__name__)


def get_class_label(class_label_file):
    class_label = open(class_label_file, 'r').readlines()
    return class_label


def get_video_time_dict(video_duration_file):
    video_time_dict = dict()
    fps_file = open(video_duration_file, 'r').readlines()
    for line in fps_file:
        contents = line.split()
        video_time_dict[contents[0]] = float(contents[-1])
    return video_time_dict


class MetricsCalculator():
    def __init__(self,
                 name='CTCN',
                 mode='train',
                 score_thresh=0.001,
                 nms_thresh=0.8,
                 sigma_thresh=0.8,
                 soft_thresh=0.006,
                 gt_label_file='',
                 class_label_file='',
                 video_duration_file=''):
        self.name = name
        self.mode = mode  # 'train', 'val', 'test'
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.sigma_thresh = sigma_thresh
        self.soft_thresh = soft_thresh
        self.class_label_file = class_label_file
        self.video_duration_file = video_duration_file
        if mode == 'test':
            lines = open(gt_label_file).readlines()
            self.gt_labels = [item.split(' ')[0] for item in lines]
            self.box_coder = BoxCoder()
        else:
            self.gt_labels = None
            self.box_coder = None
        self.reset()

    def reset(self):
        logger.info('Resetting {} metrics...'.format(self.mode))
        self.aggr_loss = 0.0
        self.aggr_loc_loss = 0.0
        self.aggr_cls_loss = 0.0
        self.aggr_batch_size = 0
        if self.mode == 'test':
            self.class_label = get_class_label(self.class_label_file)
            self.video_time_dict = get_video_time_dict(self.video_duration_file)
            self.res_detect = dict()
            self.res_detect["version"] = "VERSION 1.3"
            self.res_detect["external_data"] = {
                "uesd": False,
                "details": "none"
            }

            self.results_detect = dict()
            self.box_decode_params = {
                'score_thresh': self.score_thresh,
                'nms_thresh': self.nms_thresh,
                'sigma_thresh': self.sigma_thresh,
                'soft_thresh': self.soft_thresh
            }
            self.out_file = 'res_decode_' + str(self.score_thresh) + '_' + \
                      str(self.nms_thresh) + '_' + str(self.sigma_thresh) + \
                      '_' + str(self.soft_thresh) + '.json'

    def accumulate(self, loss, pred, label):
        cur_batch_size = 1  # iteration counter
        self.aggr_loss += np.mean(np.array(loss[0]))
        self.aggr_loc_loss += np.mean(np.array(loss[1]))
        self.aggr_cls_loss += np.mean(np.array(loss[2]))
        self.aggr_batch_size += cur_batch_size
        if self.mode == 'test':
            box_preds, label_preds, score_preds = self.box_coder.decode(
                pred[0].squeeze(), pred[1].squeeze(), **self.box_decode_params)
            fid = label.squeeze()
            fname = self.gt_labels[fid]
            logger.info("id {}, file {}, num of box preds {}:".format(
                fid, fname, len(box_preds)))
            self.results_detect[fname] = []
            for j in range(len(label_preds)):
                self.results_detect[fname].append({
                    "score": score_preds[j],
                    "label": self.class_label[label_preds[j]].strip(),
                    "segment": [
                        max(0, self.video_time_dict[fname] * box_preds[j][0] /
                            512.0), min(self.video_time_dict[fname],
                                        self.video_time_dict[fname] *
                                        box_preds[j][1] / 512.0)
                    ]
                })

    def finalize_metrics(self):
        self.avg_loss = self.aggr_loss / self.aggr_batch_size
        self.avg_loc_loss = self.aggr_loc_loss / self.aggr_batch_size
        self.avg_cls_loss = self.aggr_cls_loss / self.aggr_batch_size
        if self.mode == 'test':
            self.res_detect['results'] = self.results_detect
            with open(self.out_file, 'w') as f:
                json.dump(self.res_detect, f)
            logger.info('results has been saved into file: {}'.format(
                self.out_file))

    def get_computed_metrics(self):
        json_stats = {}
        json_stats['avg_loss'] = self.avg_loss
        json_stats['avg_loc_loss'] = self.avg_loc_loss
        json_stats['avg_cls_loss'] = self.avg_cls_loss
        return json_stats
