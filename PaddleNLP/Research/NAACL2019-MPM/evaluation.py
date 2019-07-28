#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
"""script for ensemble and evaluation."""

import os
import sys
import csv
import numpy as np
from sklearn.metrics import f1_score

label_file = sys.argv[1]
prob_file_1 = sys.argv[2]
prob_file_2 = sys.argv[3]
prob_file_3 = sys.argv[4]
prob_file_4 = sys.argv[5]


def get_labels(input_file):
    """
    get labels labels true labels file.
    """
    readers = csv.reader(open(input_file, "r"), delimiter=',')
    lines = []
    for line in readers:
        lines.append(int(line[2]))
    return lines


def get_probs(input_file):
    """
    get probs from input file.
    """
    return [float(i.strip('\n')) for i in open(input_file)]


def get_pred(probs, threshold=0.5):
    """
    get prediction from probs.
    """
    pred = []
    for p in probs:
        if p >= threshold:
            pred.append(1)
        else:
            pred.append(0)
    return pred


def vote(pred_list):
    """
    get vote result from prediction list.
    """
    pred_list = np.array(pred_list).transpose()
    preds = []
    for p in pred_list:
        counts = np.bincount(p)
        preds.append(np.argmax(counts))
    return preds


def cal_f1(preds, labels):
    """
    calculate f1 score.
    """
    return f1_score(np.array(labels), np.array(preds))


labels = get_labels(label_file)

file_list = [prob_file_1, prob_file_2, prob_file_3, prob_file_4]
pred_list = []
for f in file_list:
    pred_list.append(get_pred(get_probs(f)))

pred_ensemble = vote(pred_list)

print("all model ensemble(vote) f1: %.5f " % cal_f1(pred_ensemble, labels))
