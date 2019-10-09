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

from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import numpy as np
import datetime
import logging

logger = logging.getLogger(__name__)


class MetricsCalculator():
    def __init__(self, name, mode):
        self.name = name
        self.mode = mode  # 'train', 'val', 'test'
        self.reset()

    def reset(self):
        logger.info('Resetting {} metrics...'.format(self.mode))
        self.aggr_acc1 = 0.0
        self.aggr_acc5 = 0.0
        self.aggr_loss = 0.0
        self.aggr_batch_size = 0

    def finalize_metrics(self):
        self.avg_acc1 = self.aggr_acc1 / self.aggr_batch_size
        self.avg_acc5 = self.aggr_acc5 / self.aggr_batch_size
        self.avg_loss = self.aggr_loss / self.aggr_batch_size

    def get_computed_metrics(self):
        json_stats = {}
        json_stats['avg_loss'] = self.avg_loss
        json_stats['avg_acc1'] = self.avg_acc1
        json_stats['avg_acc5'] = self.avg_acc5
        return json_stats

    def calculate_metrics(self, loss, softmax, labels):
        accuracy1 = compute_topk_accuracy(softmax, labels, top_k=1) * 100.
        accuracy5 = compute_topk_accuracy(softmax, labels, top_k=5) * 100.
        return accuracy1, accuracy5

    def accumulate(self, loss, softmax, labels):
        cur_batch_size = softmax.shape[0]
        # if returned loss is None for e.g. test, just set loss to be 0.
        if loss is None:
            cur_loss = 0.
        else:
            cur_loss = np.mean(np.array(loss))  #
        self.aggr_batch_size += cur_batch_size
        self.aggr_loss += cur_loss * cur_batch_size

        accuracy1 = compute_topk_accuracy(softmax, labels, top_k=1) * 100.
        accuracy5 = compute_topk_accuracy(softmax, labels, top_k=5) * 100.
        self.aggr_acc1 += accuracy1 * cur_batch_size
        self.aggr_acc5 += accuracy5 * cur_batch_size

        return


# ----------------------------------------------
# other utils
# ----------------------------------------------
def compute_topk_correct_hits(top_k, preds, labels):
    '''Compute the number of corret hits'''
    batch_size = preds.shape[0]

    top_k_preds = np.zeros((batch_size, top_k), dtype=np.float32)
    for i in range(batch_size):
        top_k_preds[i, :] = np.argsort(-preds[i, :])[:top_k]

    correctness = np.zeros(batch_size, dtype=np.int32)
    for i in range(batch_size):
        if labels[i] in top_k_preds[i, :].astype(np.int32).tolist():
            correctness[i] = 1
    correct_hits = sum(correctness)

    return correct_hits


def compute_topk_accuracy(softmax, labels, top_k):

    computed_metrics = {}

    assert labels.shape[0] == softmax.shape[0], "Batch size mismatch."
    aggr_batch_size = labels.shape[0]
    aggr_top_k_correct_hits = compute_topk_correct_hits(top_k, softmax, labels)

    # normalize results
    computed_metrics = \
        float(aggr_top_k_correct_hits) / aggr_batch_size

    return computed_metrics
