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

import sys
import os
import numpy as np
import datetime
import logging
from collections import defaultdict
import pickle
import json

logger = logging.getLogger(__name__)


class MetricsCalculator():
    def __init__(self, name, mode, **metrics_args):
        """
          metrics args:
                        num_test_clips, number of clips of each video when test
                        dataset_size,   total number of videos in the dataset
                        filename_gt,    a file with each line stores the groud truth of each video
                        checkpoint_dir, dir where to store the test results
                        num_classes,    number of classes of the dataset
        """
        self.name = name
        self.mode = mode  # 'train', 'val', 'test'
        self.metrics_args = metrics_args

        self.num_test_clips = metrics_args['num_test_clips']
        self.dataset_size = metrics_args['dataset_size']
        self.filename_gt = metrics_args['filename_gt']
        self.checkpoint_dir = metrics_args['checkpoint_dir']
        self.num_classes = metrics_args['num_classes']
        self.labels_list = json.load(open(metrics_args['labels_list']))
        self.reset()

    def reset(self):
        logger.info('Resetting {} metrics...'.format(self.mode))
        self.aggr_acc1 = 0.0
        self.aggr_acc5 = 0.0
        self.aggr_loss = 0.0
        self.aggr_batch_size = 0
        self.seen_inds = defaultdict(int)
        self.results = []

    def calculate_metrics(self, loss, pred, labels):
        pass

    def accumulate(self, pred, labels):
        labels = labels.astype(int)
        labels = labels[:, 0]
        for i in range(pred.shape[0]):
            probs = pred[i, :].tolist()
            vid = labels[i]
            self.seen_inds[vid] += 1
            if self.seen_inds[vid] > self.num_test_clips:
                logger.warning('Video id {} have been seen. Skip.'.format(vid,
                                                                          ))
                continue
            save_pairs = [vid, probs]
            self.results.append(save_pairs)
            logger.info("({0} / {1}) videos".format(\
                        len(self.seen_inds), self.dataset_size))

    def accumulate_infer_results(self, pred, labels):
        for i in range(pred.shape[0]):
            vid = labels[i][0]
            probs = pred[i, :].tolist()
            self.seen_inds[vid] += 1
            if self.seen_inds[vid] > self.num_test_clips:
                logger.warning('Video id {} have been seen. Skip.'.format(vid,
                                                                          ))
                continue
            save_pairs = [vid, probs]
            self.results.append(save_pairs)

    def finalize_metrics(self):
        if self.filename_gt is not None:
            evaluate_results(self.results, self.filename_gt, self.dataset_size, \
                             self.num_classes, self.num_test_clips)

    def finalize_infer_metrics(self):
        evaluate_infer_results(self.results, self.num_classes,
                               self.num_test_clips, self.labels_list)


def read_groundtruth(filename_gt):
    f = open(filename_gt, 'r')
    labels = []
    for line in f:
        rows = line.split()
        labels.append(int(rows[1]))
    f.close()
    return labels


def evaluate_results(results, filename_gt, test_dataset_size, num_classes,
                     num_test_clips):
    gt_labels = read_groundtruth(filename_gt)
    sample_num = test_dataset_size
    class_num = num_classes
    sample_video_times = num_test_clips
    counts = np.zeros(sample_num, dtype=np.int32)
    probs = np.zeros((sample_num, class_num))

    assert (len(gt_labels) == sample_num), \
             "the number of gt_labels({}) should be the same with sample_num({})".format(
                         len(gt_labels), sample_num)
    """
    clip_accuracy: the (e.g.) 10*19761 clips' average accuracy
    clip1_accuracy: the 1st clip's accuracy (starting from frame 0)
    """
    clip_accuracy = 0
    clip1_accuracy = 0
    clip1_count = 0
    seen_inds = defaultdict(int)

    # evaluate
    for entry in results:
        vid = entry[0]
        prob = np.array(entry[1])
        probs[vid] += prob[0:class_num]
        counts[vid] += 1

        idx = prob.argmax()
        if idx == gt_labels[vid]:
            # clip accuracy
            clip_accuracy += 1

        # clip1 accuracy
        seen_inds[vid] += 1
        if seen_inds[vid] == 1:
            clip1_count += 1
            if idx == gt_labels[vid]:
                clip1_accuracy += 1

    # sanity checkcnt = 0
    max_clips = 0
    min_clips = sys.maxsize
    count_empty = 0
    count_corrupted = 0
    for i in range(sample_num):
        max_clips = max(max_clips, counts[i])
        min_clips = min(min_clips, counts[i])
        if counts[i] != sample_video_times:
            count_corrupted += 1
            logger.warning('Id: {} count: {}'.format(i, counts[i]))
        if counts[i] == 0:
            count_empty += 1

    logger.info('Num of empty videos: {}'.format(count_empty))
    logger.info('Num of corrupted videos: {}'.format(count_corrupted))
    logger.info('Max num of clips in a video: {}'.format(max_clips))
    logger.info('Min num of clips in a video: {}'.format(min_clips))

    # clip1 accuracy for sanity (# print clip1 first as it is lowest)
    logger.info('Clip1 accuracy: {:.2f} percent ({}/{})'.format(
        100. * clip1_accuracy / clip1_count, clip1_accuracy, clip1_count))

    # clip accuracy for sanity
    logger.info('Clip accuracy: {:.2f} percent ({}/{})'.format(
        100. * clip_accuracy / len(results), clip_accuracy, len(results)))

    # compute accuracy
    accuracy = 0
    accuracy_top5 = 0
    for i in range(sample_num):
        prob = probs[i]

        # top-1
        idx = prob.argmax()
        if idx == gt_labels[i] and counts[i] > 0:
            accuracy = accuracy + 1

        ids = np.argsort(prob)[::-1]
        for j in range(5):
            if ids[j] == gt_labels[i] and counts[i] > 0:
                accuracy_top5 = accuracy_top5 + 1
                break

    accuracy = float(accuracy) / float(sample_num)
    accuracy_top5 = float(accuracy_top5) / float(sample_num)

    logger.info('-' * 80)
    logger.info('top-1 accuracy: {:.2f} percent'.format(accuracy * 100))
    logger.info('top-5 accuracy: {:.2f} percent'.format(accuracy_top5 * 100))
    logger.info('-' * 80)

    return


def evaluate_infer_results(results, num_classes, num_test_clips, labels_list):
    probs = {}
    counts = {}
    for entry in results:
        vid = entry[0]
        pred = entry[1]
        if vid in probs.keys():
            assert vid in counts.keys(
            ), "If vid in probs, it should be in counts"
            probs[vid] = (probs[vid] * counts[vid] + pred) / (counts[vid] + 1)
            counts[vid] += 1
        else:
            probs[vid] = np.copy(pred)
            counts[vid] = 1

    topk = 20

    for vid in probs.keys():
        pred = probs[vid]
        sorted_inds = np.argsort(pred)[::-1]
        topk_inds = sorted_inds[:topk]
        logger.info('video {}, topk({}) preds: \n'.format(vid, topk))
        for ind in topk_inds:
            logger.info('\t    class: {},  probability  {} \n'.format(
                labels_list[ind], pred[ind]))
