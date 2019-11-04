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
import operator

logger = logging.getLogger(__name__)


class MetricsCalculator():
    def __init__(
            self,
            name='TALL',
            mode='train', ):
        self.name = name
        self.mode = mode  # 'train', 'valid', 'test', 'infer'
        self.reset()

    def reset(self):
        logger.info('Resetting {} metrics...'.format(self.mode))
        if (self.mode == 'train') or (self.mode == 'valid'):
            self.aggr_loss = 0.0
        elif (self.mode == 'test') or (self.mode == 'infer'):
            self.result_dict = dict()
            self.save_res = dict()
            self.out_file = self.name + '_' + self.mode + '_res_' + '.json'

    def nms_temporal(self, x1, x2, sim, overlap):
        pick = []
        assert len(x1) == len(sim)
        assert len(x2) == len(sim)
        if len(x1) == 0:
            return pick

        union = list(map(operator.sub, x2, x1))  # union = x2-x1

        I = [i[0] for i in sorted(
            enumerate(sim), key=lambda x: x[1])]  # sort and get index

        while len(I) > 0:
            i = I[-1]
            pick.append(i)

            xx1 = [max(x1[i], x1[j]) for j in I[:-1]]
            xx2 = [min(x2[i], x2[j]) for j in I[:-1]]
            inter = [max(0.0, k2 - k1) for k1, k2 in zip(xx1, xx2)]
            o = [
                inter[u] / (union[i] + union[I[u]] - inter[u])
                for u in range(len(I) - 1)
            ]
            I_new = []
            for j in range(len(o)):
                if o[j] <= overlap:
                    I_new.append(I[j])
            I = I_new
        return pick

    def calculate_IoU(self, i0, i1):
        # calculate temporal intersection over union
        union = (min(i0[0], i1[0]), max(i0[1], i1[1]))
        inter = (max(i0[0], i1[0]), min(i0[1], i1[1]))
        iou = 1.0 * (inter[1] - inter[0]) / (union[1] - union[0])
        return iou

    def compute_IoU_recall_top_n_forreg(self, top_n, iou_thresh,
                                        sentence_image_mat,
                                        sentence_image_reg_mat, sclips):
        correct_num = 0.0
        for k in range(sentence_image_mat.shape[0]):
            gt = sclips[k]
            gt_start = float(gt.split("_")[1])
            gt_end = float(gt.split("_")[2])
            sim_v = [v for v in sentence_image_mat[k]]
            starts = [s for s in sentence_image_reg_mat[k, :, 0]]
            ends = [e for e in sentence_image_reg_mat[k, :, 1]]
            picks = self.nms_temporal(starts, ends, sim_v, iou_thresh - 0.05)
            if top_n < len(picks):
                picks = picks[0:top_n]
            for index in picks:
                pred_start = sentence_image_reg_mat[k, index, 0]
                pred_end = sentence_image_reg_mat[k, index, 1]
                iou = self.calculate_IoU((gt_start, gt_end),
                                         (pred_start, pred_end))
                if iou >= iou_thresh:
                    correct_num += 1
                    break
        return correct_num

    def accumulate(self, fetch_list):
        if self.mode == 'valid':
            loss = fetch_list[0]
            self.aggr_loss += np.mean(np.array(loss))
        elif (self.mode == 'test') or (self.mode == 'infer'):
            outputs = fetch_list[0]
            b_start = [item[0] for item in fetch_list[1]]
            b_end = [item[1] for item in fetch_list[1]]
            b_k = [item[2] for item in fetch_list[1]]
            b_t = [item[3] for item in fetch_list[1]]
            b_movie_clip_sentences = [item[4] for item in fetch_list[1]]
            b_movie_clip_featmaps = [item[5] for item in fetch_list[1]]
            b_movie_name = [item[6] for item in fetch_list[1]]

            batch_size = len(b_start)
            for i in range(batch_size):
                start = b_start[i]
                end = b_end[i]
                k = b_k[i]
                t = b_t[i]
                movie_clip_sentences = b_movie_clip_sentences[i]
                movie_clip_featmaps = b_movie_clip_featmaps[i]
                movie_name = b_movie_name[i]

                item_res = [outputs, start, end, k, t]

                if movie_name not in self.result_dict.keys():
                    self.result_dict[movie_name] = []
                    self.result_dict[movie_name].append(movie_clip_sentences)
                    self.result_dict[movie_name].append(movie_clip_featmaps)

                self.result_dict[movie_name].append(item_res)

    def accumulate_infer_results(self, fetch_list):
        # the same as test
        pass

    def finalize_metrics(self, savedir):
        # init
        IoU_thresh = [0.1, 0.3, 0.5, 0.7]
        all_correct_num_10 = [0.0] * 5
        all_correct_num_5 = [0.0] * 5
        all_correct_num_1 = [0.0] * 5
        all_retrievd = 0.0

        idx = 0
        all_number = len(self.result_dict)
        for movie_name in self.result_dict.keys():
            idx += 1
            logger.info('{} / {}'.format('%d' % idx, '%d' % all_number))

            movie_clip_sentences = self.result_dict[movie_name][0]
            movie_clip_featmaps = self.result_dict[movie_name][1]

            ls = len(movie_clip_sentences)
            lf = len(movie_clip_featmaps)
            sentence_image_mat = np.zeros([ls, lf])
            sentence_image_reg_mat = np.zeros([ls, lf, 2])

            movie_res = self.result_dict[movie_name][2:]
            for item_res in movie_res:
                outputs, start, end, k, t = item_res

                outputs = np.squeeze(outputs)
                sentence_image_mat[k, t] = outputs[0]
                reg_end = end + outputs[2]
                reg_start = start + outputs[1]

                sentence_image_reg_mat[k, t, 0] = reg_start
                sentence_image_reg_mat[k, t, 1] = reg_end

            sclips = [b[0] for b in movie_clip_sentences]

            for i in range(len(IoU_thresh)):
                IoU = IoU_thresh[i]
                correct_num_10 = self.compute_IoU_recall_top_n_forreg(
                    10, IoU, sentence_image_mat, sentence_image_reg_mat, sclips)
                correct_num_5 = self.compute_IoU_recall_top_n_forreg(
                    5, IoU, sentence_image_mat, sentence_image_reg_mat, sclips)
                correct_num_1 = self.compute_IoU_recall_top_n_forreg(
                    1, IoU, sentence_image_mat, sentence_image_reg_mat, sclips)

                logger.info(
                    movie_name +
                    " IoU= {}, R@10: {}; IoU= {}, R@5: {}; IoU= {}, R@1: {}".
                    format('%s' % str(IoU), '%s' % str(correct_num_10 / len(
                        sclips)), '%s' % str(IoU), '%s' % str(
                            correct_num_5 / len(sclips)), '%s' % str(IoU), '%s'
                           % str(correct_num_1 / len(sclips))))

                all_correct_num_10[i] += correct_num_10
                all_correct_num_5[i] += correct_num_5
                all_correct_num_1[i] += correct_num_1

            all_retrievd += len(sclips)

        for j in range(len(IoU_thresh)):
            logger.info(
                " IoU= {}, R@10: {}; IoU= {}, R@5: {}; IoU= {}, R@1: {}".format(
                    '%s' % str(IoU_thresh[j]), '%s' % str(all_correct_num_10[
                        j] / all_retrievd), '%s' % str(IoU_thresh[j]), '%s' %
                    str(all_correct_num_5[j] / all_retrievd), '%s' % str(
                        IoU_thresh[j]), '%s' % str(all_correct_num_1[j] /
                                                   all_retrievd)))

        self.R1_IOU5 = all_correct_num_1[2] / all_retrievd
        self.R5_IOU5 = all_correct_num_5[2] / all_retrievd

        self.save_res["best_R1_IOU5"] = self.R1_IOU5
        self.save_res["best_R5_IOU5"] = self.R5_IOU5

        self.filepath = os.path.join(savedir, self.out_file)
        with open(self.filepath, 'w') as f:
            f.write(
                json.dumps(
                    {
                        'version': 'VERSION 1.0',
                        'results': self.save_res,
                        'external_data': {}
                    },
                    indent=2))
            logger.info('results has been saved into file: {}'.format(
                self.filepath))

    def finalize_infer_metrics(self, savedir):
        idx = 0
        all_number = len(self.result_dict)
        res = dict()
        for movie_name in self.result_dict.keys():
            res[movie_name] = []
            idx += 1
            logger.info('{} / {}'.format('%d' % idx, '%d' % all_number))

            movie_clip_sentences = self.result_dict[movie_name][0]
            movie_clip_featmaps = self.result_dict[movie_name][1]

            ls = len(movie_clip_sentences)
            lf = len(movie_clip_featmaps)
            sentence_image_mat = np.zeros([ls, lf])
            sentence_image_reg_mat = np.zeros([ls, lf, 2])

            movie_res = self.result_dict[movie_name][2:]
            for item_res in movie_res:
                outputs, start, end, k, t = item_res

                outputs = np.squeeze(outputs)
                sentence_image_mat[k, t] = outputs[0]
                reg_end = end + outputs[2]
                reg_start = start + outputs[1]

                sentence_image_reg_mat[k, t, 0] = reg_start
                sentence_image_reg_mat[k, t, 1] = reg_end

            sclips = [b[0] for b in movie_clip_sentences]
            IoU = 0.5  #pre-define
            for k in range(sentence_image_mat.shape[0]):
                #ground_truth for compare
                gt = sclips[k]
                gt_start = float(gt.split("_")[1])
                gt_end = float(gt.split("_")[2])

                sim_v = [v for v in sentence_image_mat[k]]
                starts = [s for s in sentence_image_reg_mat[k, :, 0]]
                ends = [e for e in sentence_image_reg_mat[k, :, 1]]
                picks = self.nms_temporal(starts, ends, sim_v, IoU - 0.05)

                if 1 < len(picks):  #top1
                    picks = picks[0:1]

                for index in picks:
                    pred_start = sentence_image_reg_mat[k, index, 0]
                    pred_end = sentence_image_reg_mat[k, index, 1]
                    res[movie_name].append((k, pred_start, pred_end))

                logger.info(
                    'movie_name: {}, sentence_id: {}, pred_start_time: {}, pred_end_time: {}, gt_start_time: {}, gt_end_time: {}'.
                    format('%s' % movie_name, '%s' % str(k), '%s' % str(
                        pred_start), '%s' % str(pred_end), '%s' % str(gt_start),
                           '%s' % str(gt_end)))

        self.filepath = os.path.join(savedir, self.out_file)
        with open(self.filepath, 'w') as f:
            f.write(
                json.dumps(
                    {
                        'version': 'VERSION 1.0',
                        'results': res,
                        'external_data': {}
                    },
                    indent=2))
            logger.info('results has been saved into file: {}'.format(
                self.filepath))

    def get_computed_metrics(self):
        return self.save_res
