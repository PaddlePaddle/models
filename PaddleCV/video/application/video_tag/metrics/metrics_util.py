#  Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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

import logging

import os
import io
import numpy as np
import json
from metrics.youtube8m import eval_util as youtube8m_metrics
from metrics.kinetics import accuracy_metrics as kinetics_metrics

logger = logging.getLogger(__name__)


class Metrics(object):
    def __init__(self, name, mode, metrics_args):
        """Not implemented"""
        pass

    def calculate_and_log_out(self, fetch_list, info=''):
        """Not implemented"""
        pass

    def accumulate(self, fetch_list, info=''):
        """Not implemented"""
        pass

    def finalize_and_log_out(self, info='', savedir='./'):
        """Not implemented"""
        pass

    def reset(self):
        """Not implemented"""
        pass


class Youtube8mMetrics(Metrics):
    def __init__(self, name, mode, metrics_args):
        self.name = name
        self.mode = mode
        self.num_classes = metrics_args['MODEL']['num_classes']
        self.topk = metrics_args['MODEL']['topk']
        self.calculator = youtube8m_metrics.EvaluationMetrics(self.num_classes,
                                                              self.topk)
        if self.mode == 'infer':
            self.infer_results = []

    def calculate_and_log_out(self, fetch_list, info=''):
        loss = np.mean(np.array(fetch_list[0]))
        pred = np.array(fetch_list[1])
        label = np.array(fetch_list[2])
        hit_at_one = youtube8m_metrics.calculate_hit_at_one(pred, label)
        perr = youtube8m_metrics.calculate_precision_at_equal_recall_rate(pred,
                                                                          label)
        gap = youtube8m_metrics.calculate_gap(pred, label)
        logger.info(info + ' , loss = {0}, Hit@1 = {1}, PERR = {2}, GAP = {3}'.format(\
                     '%.6f' % loss, '%.2f' % hit_at_one, '%.2f' % perr, '%.2f' % gap))

    def accumulate(self, fetch_list, info=''):
        if self.mode == 'infer':
            predictions = np.array(fetch_list[0])
            video_id = fetch_list[1]
            for i in range(len(predictions)):
                topk_inds = predictions[i].argsort()[0 - self.topk:]
                topk_inds = topk_inds[::-1]
                preds = predictions[i][topk_inds]
                self.infer_results.append(
                    (video_id[i], topk_inds.tolist(), preds.tolist()))
        else:
            loss = np.array(fetch_list[0])
            pred = np.array(fetch_list[1])
            label = np.array(fetch_list[2])
            self.calculator.accumulate(loss, pred, label)

    def finalize_and_log_out(self,
                             info='',
                             savedir='./data/results',
                             label_file='./label_3396.txt'):
        if self.mode == 'infer':
            for index, item in enumerate(self.infer_results):
                video_id = item[0]
                print('[========video_id [ {} ] , topk({}) preds: ========]\n'.
                      format(video_id, self.topk))

                f = io.open(label_file, "r", encoding="utf-8")
                fl = f.readlines()
                res_list = []
                res_list.append(video_id)
                for i in range(len(item[1])):
                    class_id = item[1][i]
                    class_prob = item[2][i]
                    class_name = fl[class_id].split('\n')[0]
                    print('class_id: {},'.format(class_id), 'class_name:',
                          class_name,
                          ',  probability:  {} \n'.format(class_prob))
                    save_dict = {
                        "'class_id": class_id,
                        "class_name": class_name,
                        "probability": class_prob
                    }
                    res_list.append(save_dict)

                # save infer result into output dir
                with io.open(
                        os.path.join(savedir, 'result' + str(index) + '.json'),
                        'w',
                        encoding='utf-8') as f:
                    f.write(json.dumps(res_list, ensure_ascii=False))
        else:
            epoch_info_dict = self.calculator.get()
            logger.info(info + '\tavg_hit_at_one: {0},\tavg_perr: {1},\tavg_loss :{2},\taps: {3},\tgap:{4}'\
                     .format(epoch_info_dict['avg_hit_at_one'], epoch_info_dict['avg_perr'], \
                             epoch_info_dict['avg_loss'], epoch_info_dict['aps'], epoch_info_dict['gap']))

    def reset(self):
        self.calculator.clear()
        if self.mode == 'infer':
            self.infer_results = []


class Kinetics400Metrics(Metrics):
    def __init__(self, name, mode, metrics_args):
        self.name = name
        self.mode = mode
        self.topk = metrics_args['MODEL']['topk']
        self.calculator = kinetics_metrics.MetricsCalculator(name, mode.lower())
        if self.mode == 'infer':
            self.infer_results = []

    def calculate_and_log_out(self, fetch_list, info=''):
        if len(fetch_list) == 3:
            loss = fetch_list[0]
            loss = np.mean(np.array(loss))
            pred = np.array(fetch_list[1])
            label = np.array(fetch_list[2])
        else:
            loss = 0.
            pred = np.array(fetch_list[0])
            label = np.array(fetch_list[1])
        acc1, acc5 = self.calculator.calculate_metrics(loss, pred, label)
        logger.info(info + '\tLoss: {},\ttop1_acc: {}, \ttop5_acc: {}'.format('%.6f' % loss, \
                       '%.2f' % acc1, '%.2f' % acc5))
        return loss

    def accumulate(self, fetch_list, info=''):
        if self.mode == 'infer':
            predictions = np.array(fetch_list[0])
            video_id = fetch_list[1]
            for i in range(len(predictions)):
                topk_inds = predictions[i].argsort()[0 - self.topk:]
                topk_inds = topk_inds[::-1]
                preds = predictions[i][topk_inds]
                self.infer_results.append(
                    (video_id[i], topk_inds.tolist(), preds.tolist()))
        else:
            if len(fetch_list) == 3:
                loss = fetch_list[0]
                loss = np.mean(np.array(loss))
                pred = np.array(fetch_list[1])
                label = np.array(fetch_list[2])
            else:
                loss = 0.
                pred = np.array(fetch_list[0])
                label = np.array(fetch_list[1])
            self.calculator.accumulate(loss, pred, label)

    def finalize_and_log_out(self,
                             info='',
                             savedir='./data/results',
                             label_file='./label_3396.txt'):
        if self.mode == 'infer':
            for index, item in enumerate(self.infer_results):
                video_id = item[0]
                print('[========video_id [ {} ] , topk({}) preds: ========]\n'.
                      format(video_id, self.topk))

                f = io.open(label_file, "r", encoding="utf-8")
                fl = f.readlines()
                res_list = []
                res_list.append(video_id)
                for i in range(len(item[1])):
                    class_id = item[1][i]
                    class_prob = item[2][i]
                    class_name = fl[class_id].split('\n')[0]
                    print('class_id: {},'.format(class_id), 'class_name:',
                          class_name,
                          ',  probability:  {} \n'.format(class_prob))
                    save_dict = {
                        "'class_id": class_id,
                        "class_name": class_name,
                        "probability": class_prob
                    }
                    res_list.append(save_dict)

                # save infer result into output dir
                with io.open(
                        os.path.join(savedir, 'result' + str(index) + '.json'),
                        'w',
                        encoding='utf-8') as f:
                    f.write(json.dumps(res_list, ensure_ascii=False))
        else:
            self.calculator.finalize_metrics()
            metrics_dict = self.calculator.get_computed_metrics()
            loss = metrics_dict['avg_loss']
            acc1 = metrics_dict['avg_acc1']
            acc5 = metrics_dict['avg_acc5']
            logger.info(info + '\tLoss: {},\ttop1_acc: {}, \ttop5_acc: {}'.format('%.6f' % loss, \
                       '%.2f' % acc1, '%.2f' % acc5))

    def reset(self):
        self.calculator.reset()
        if self.mode == 'infer':
            self.infer_results = []


class MetricsZoo(object):
    def __init__(self):
        self.metrics_zoo = {}

    def regist(self, name, metrics):
        assert metrics.__base__ == Metrics, "Unknow model type {}".format(
            type(metrics))
        self.metrics_zoo[name] = metrics

    def get(self, name, mode, cfg):
        for k, v in self.metrics_zoo.items():
            if k == name:
                return v(name, mode, cfg)
        raise MetricsNotFoundError(name, self.metrics_zoo.keys())


# singleton metrics_zoo
metrics_zoo = MetricsZoo()


def regist_metrics(name, metrics):
    metrics_zoo.regist(name, metrics)


def get_metrics(name, mode, cfg):
    return metrics_zoo.get(name, mode, cfg)


# sort by alphabet
regist_metrics("ATTENTIONLSTM", Youtube8mMetrics)
regist_metrics("TSN", Kinetics400Metrics)
