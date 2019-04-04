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

import logging

import numpy as np
from metrics.youtube8m import eval_util as youtube8m_metrics
from metrics.kinetics import accuracy_metrics as kinetics_metrics
from metrics.multicrop_test import multicrop_test_metrics as multicrop_test_metrics

logger = logging.getLogger(__name__)


class Metrics(object):
    def __init__(self, name, mode, metrics_args):
        """Not implemented"""
        pass

    def calculate_and_log_out(self, loss, pred, label, info=''):
        """Not implemented"""
        pass

    def accumulate(self, loss, pred, label, info=''):
        """Not implemented"""
        pass

    def finalize_and_log_out(self, info=''):
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

    def calculate_and_log_out(self, loss, pred, label, info=''):
        loss = np.mean(np.array(loss))
        hit_at_one = youtube8m_metrics.calculate_hit_at_one(pred, label)
        perr = youtube8m_metrics.calculate_precision_at_equal_recall_rate(pred,
                                                                          label)
        gap = youtube8m_metrics.calculate_gap(pred, label)
        logger.info(info + ' , loss = {0}, Hit@1 = {1}, PERR = {2}, GAP = {3}'.format(\
                     '%.6f' % loss, '%.2f' % hit_at_one, '%.2f' % perr, '%.2f' % gap))

    def accumulate(self, loss, pred, label, info=''):
        self.calculator.accumulate(loss, pred, label)

    def finalize_and_log_out(self, info=''):
        epoch_info_dict = self.calculator.get()
        logger.info(info + '\tavg_hit_at_one: {0},\tavg_perr: {1},\tavg_loss :{2},\taps: {3},\tgap:{4}'\
                     .format(epoch_info_dict['avg_hit_at_one'], epoch_info_dict['avg_perr'], \
                             epoch_info_dict['avg_loss'], epoch_info_dict['aps'], epoch_info_dict['gap']))

    def reset(self):
        self.calculator.clear()


class Kinetics400Metrics(Metrics):
    def __init__(self, name, mode, metrics_args):
        self.name = name
        self.mode = mode
        self.calculator = kinetics_metrics.MetricsCalculator(name, mode.lower())

    def calculate_and_log_out(self, loss, pred, label, info=''):
        if loss is not None:
            loss = np.mean(np.array(loss))
        else:
            loss = 0.
        acc1, acc5 = self.calculator.calculate_metrics(loss, pred, label)
        logger.info(info + '\tLoss: {},\ttop1_acc: {}, \ttop5_acc: {}'.format('%.6f' % loss, \
                       '%.2f' % acc1, '%.2f' % acc5))

    def accumulate(self, loss, pred, label, info=''):
        self.calculator.accumulate(loss, pred, label)

    def finalize_and_log_out(self, info=''):
        self.calculator.finalize_metrics()
        metrics_dict = self.calculator.get_computed_metrics()
        loss = metrics_dict['avg_loss']
        acc1 = metrics_dict['avg_acc1']
        acc5 = metrics_dict['avg_acc5']
        logger.info(info + '\tLoss: {},\ttop1_acc: {}, \ttop5_acc: {}'.format('%.6f' % loss, \
                       '%.2f' % acc1, '%.2f' % acc5))

    def reset(self):
        self.calculator.reset()


class MulticropMetrics(Metrics):
    def __init__(self, name, mode, metrics_args):
        self.name = name
        self.mode = mode
        if mode == 'test':
            args = {}
            args['num_test_clips'] = metrics_args.TEST.num_test_clips
            args['dataset_size'] = metrics_args.TEST.dataset_size
            args['filename_gt'] = metrics_args.TEST.filename_gt
            args['checkpoint_dir'] = metrics_args.TEST.checkpoint_dir
            args['num_classes'] = metrics_args.MODEL.num_classes
            self.calculator = multicrop_test_metrics.MetricsCalculator(
                name, mode.lower(), **args)
        else:
            self.calculator = kinetics_metrics.MetricsCalculator(name,
                                                                 mode.lower())

    def calculate_and_log_out(self, loss, pred, label, info=''):
        if self.mode == 'test':
            pass
        else:
            if loss is not None:
                loss = np.mean(np.array(loss))
            else:
                loss = 0.
            acc1, acc5 = self.calculator.calculate_metrics(loss, pred, label)
            logger.info(info + '\tLoss: {},\ttop1_acc: {}, \ttop5_acc: {}'.format('%.6f' % loss, \
                                   '%.2f' % acc1, '%.2f' % acc5))

    def accumulate(self, loss, pred, label):
        self.calculator.accumulate(loss, pred, label)

    def finalize_and_log_out(self, info=''):
        if self.mode == 'test':
            self.calculator.finalize_metrics()
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
regist_metrics("ATTENTIONCLUSTER", Youtube8mMetrics)
regist_metrics("ATTENTIONLSTM", Youtube8mMetrics)
regist_metrics("NEXTVLAD", Youtube8mMetrics)
regist_metrics("NONLOCAL", MulticropMetrics)
regist_metrics("TSM", Kinetics400Metrics)
regist_metrics("TSN", Kinetics400Metrics)
regist_metrics("STNET", Kinetics400Metrics)
