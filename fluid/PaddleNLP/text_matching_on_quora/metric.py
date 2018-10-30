# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved
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

"""
This Module defines evaluate metrics for classification tasks
"""


def accuracy(y_pred, label):
    """
    define correct: the top 1 class in y_pred is the same as y_true
    """
    y_pred = np.squeeze(y_pred)
    y_pred_idx = np.argmax(y_pred, axis=1)
    return 1.0 * np.sum(y_pred_idx == label) / label.shape[0]

def precision(y_pred, label):
    """
    """
    y_pred = np.squeeze(y_pred)
    y_pred_idx = np.argmax(y_pred, axis=1)
    positive_cross_count = 0
    positive_pred_count = sum(y_pred_idx)
    for i in range(len(y_pred_idx)):
        if y_pred_idx[i] == 1 and label[i] == 1:
            positive_cross_count += 1
    if positive_pred_count == 0:
        return 0
    return positive_cross_count / (1.0 * positive_pred_count)


def recall(y_pred, label):
    """
    """
    y_pred = np.squeeze(y_pred)
    y_pred_idx = np.argmax(y_pred, axis=1)
    positive_real_count = sum(label)
    positive_cross_count = 0
    for i in range(len(y_pred_idx)):
        if y_pred_idx[i] == 1 and label[i] == 1:
            positive_cross_count += 1
    if positive_real_count == 0:
        return 0
    return positive_cross_count / (1.0 * positive_real_count)


def f1(y_pred, label):
    """
    """
    prec = precision(y_pred, label)
    rec = recall(y_pred, label)
    if prec == 0 and rec == 0:
        return 0
    return 2.0 * prec * rec / (prec + rec)

def accuracy_with_threshold(y_pred, label, threshold=0.5):
    """
    define correct: the y_true class's prob in y_pred is bigger than threshold
    when threshold is 0.5, This fuction is equal to accuracy
    """
    y_pred = np.squeeze(y_pred)
    y_pred_idx = (y_pred[:, 1] > threshold).astype(int)
    return 1.0 * np.sum(y_pred_idx == label) / label.shape[0]

def cal_all_metric(y_pred, label, metric_type):
    """
    """
    metric_res = []
    for metric_name in metric_type:
        if metric_name == 'accuracy_with_threshold':
            metric_res.append((metric_name, accuracy_with_threshold(y_pred, label, threshold=0.3)))
        elif metric_name == 'accuracy':
            metric_res.append((metric_name, accuracy(y_pred, label)))
        elif metric_name == 'precision':
            metric_res.append((metric_name, precision(y_pred, label)))
        elif metric_name == 'recall':
            metric_res.append((metric_name, recall(y_pred, label)))
        elif metric_name == 'f1':
            metric_res.append((metric_name, f1(y_pred, label)))
        else:
            print("Unknown metric type: ", metric_name)
            exit()
    return metric_res
