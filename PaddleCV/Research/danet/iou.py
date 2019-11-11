# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np


class IOUMetric:

    def __init__(self, num_classes):
        self.num_classes = num_classes+1
        self.hist = np.zeros((num_classes+1, num_classes+1))
        
    def _fast_hist(self, label_pred, label_true):
        mask = (label_true >= 0) & (label_true < self.num_classes)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) +
            label_pred[mask], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        return hist

    def add_batch(self, predictions, gts):
        # gts = BHW
        # predictions = BHW
        if isinstance(gts, np.ndarray):
            gts_ig = (gts == 255).astype(np.int32)
            gts_nig = (gts != 255).astype(np.int32)
            # print(predictions)
            gts[gts == 255] = self.num_classes-1  # 19
            predictions = gts_nig * predictions + gts_ig * (self.num_classes-1)
            # print(predictions)
        for lp, lt in zip(predictions, gts):
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())

    def evaluate(self):
        acc = np.diag(self.hist).sum() / self.hist.sum()
        acc_cls = np.nanmean(np.diag(self.hist) / self.hist.sum(axis=1))
        iu = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist))
        mean_iu = np.nanmean(iu)
        freq = self.hist.sum(axis=1) / self.hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        kappa = (self.hist.sum() * np.diag(self.hist).sum() - (self.hist.sum(axis=0) * self.hist.sum(axis=1)).sum()) / (
                self.hist.sum() ** 2 - (self.hist.sum(axis=0) * self.hist.sum(axis=1)).sum())
        return acc, acc_cls, iu, mean_iu, fwavacc, kappa

    def evaluate_kappa(self):
        kappa = (self.hist.sum() * np.diag(self.hist).sum() - (self.hist.sum(axis=0) * self.hist.sum(axis=1)).sum()) / (
                self.hist.sum() ** 2 - (self.hist.sum(axis=0) * self.hist.sum(axis=1)).sum())
        return kappa

    def evaluate_iou_kappa(self):
        iu = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist))
        mean_iu = np.nanmean(iu)
        kappa = (self.hist.sum() * np.diag(self.hist).sum() - (self.hist.sum(axis=0) * self.hist.sum(axis=1)).sum()) / (
                self.hist.sum() ** 2 - (self.hist.sum(axis=0) * self.hist.sum(axis=1)).sum())
        return mean_iu, kappa

    def evaluate_iu(self):
        iu = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist))
        return iu

