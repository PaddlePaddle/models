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
import os

from models.ctcn.ctcn_utils import BoxCoder

logger = logging.getLogger(__name__)


class MetricsCalculator():
    def __init__(self, name='ETS', mode='train', dict_file=''):
        self.name = name
        self.mode = mode  # 'train', 'valid', 'test', 'infer'
        self.dict_file = dict_file
        self.reset()

    def reset(self):
        logger.info('Resetting {} metrics...'.format(self.mode))
        self.aggr_batch_size = 0
        if (self.mode == 'train') or (self.mode == 'valid'):
            self.aggr_loss = 0.0
        elif (self.mode == 'test') or (self.mode == 'infer'):
            self.result_dict = dict()
            self.out_file = self.name + '_' + self.mode + '_res_' + '.json'

    def accumulate(self, fetch_list):
        if self.mode == 'valid':
            loss = fetch_list[0]
            self.aggr_loss += np.mean(np.array(loss))
        elif (self.mode == 'test') or (self.mode == 'infer'):
            seq_ids = fetch_list[0]
            seq_scores = fetch_list[1]
            b_vid = [item[0] for item in fetch_list[2]]
            b_stime = [item[1] for item in fetch_list[2]]
            b_etime = [item[2] for item in fetch_list[2]]

            # for test and inference, batch size=1
            vid = b_vid[0]
            stime = b_stime[0]
            etime = b_etime[0]

            #get idx_to_word
            self.idx_to_word = dict()
            with open(self.dict_file, 'r') as f:
                for i, line in enumerate(f):
                    self.idx_to_word[i] = line.strip().split()[0]

            for i in range(len(seq_ids.lod()[0]) - 1):
                start = seq_ids.lod()[0][i]
                end = seq_ids.lod()[0][i + 1]
                for j in range(end - start)[:1]:
                    sub_start = seq_ids.lod()[1][start + j]
                    sub_end = seq_ids.lod()[1][start + j + 1]
                    sent = " ".join([
                        self.idx_to_word[idx]
                        for idx in np.array(seq_ids)[sub_start:sub_end][1:-1]
                    ])
                    if vid not in self.result_dict:
                        self.result_dict[vid] = [{
                            'timestamp': [stime, etime],
                            'sentence': sent
                        }]
                    else:
                        self.result_dict[vid].append({
                            'timestamp': [stime, etime],
                            'sentence': sent
                        })

    def accumulate_infer_results(self, fetch_list):
        # the same as test
        pass

    def finalize_metrics(self, savedir):
        self.filepath = os.path.join(savedir, self.out_file)
        with open(self.filepath, 'w') as f:
            f.write(
                json.dumps(
                    {
                        'version': 'VERSION 1.0',
                        'results': self.result_dict,
                        'external_data': {}
                    },
                    indent=2))
            logger.info('results has been saved into file: {}'.format(
                self.filepath))

    def finalize_infer_metrics(self, savedir):
        # the same as test
        pass

    def get_computed_metrics(self):
        pass
