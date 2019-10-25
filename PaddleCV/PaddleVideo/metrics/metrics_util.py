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
import json
from metrics.youtube8m import eval_util as youtube8m_metrics
from metrics.kinetics import accuracy_metrics as kinetics_metrics
from metrics.multicrop_test import multicrop_test_metrics as multicrop_test_metrics
from metrics.detections import detection_metrics as detection_metrics
from metrics.bmn_metrics import bmn_proposal_metrics as bmn_proposal_metrics
from metrics.bsn_metrics import bsn_tem_metrics as bsn_tem_metrics
from metrics.bsn_metrics import bsn_pem_metrics as bsn_pem_metrics
from metrics.ets_metrics import ets_metrics as ets_metrics
from metrics.tall_metrics import tall_metrics as tall_metrics

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

    def finalize_and_log_out(self, info='', savedir='./'):
        if self.mode == 'infer':
            for item in self.infer_results:
                logger.info('video_id {} , topk({}) preds: \n'.format(item[
                    0], self.topk))
                for i in range(len(item[1])):
                    logger.info('\t    class: {},  probability  {} \n'.format(
                        item[1][i], item[2][i]))
            # save infer result into output dir
            #json.dump(self.infer_results, xxxx)

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
            self.kinetics_labels = metrics_args['INFER']['kinetics_labels']
            self.labels_list = json.load(open(self.kinetics_labels))

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

    def finalize_and_log_out(self, info='', savedir='./'):
        if self.mode == 'infer':
            for item in self.infer_results:
                logger.info('video_id {} , topk({}) preds: \n'.format(item[
                    0], self.topk))
                for i in range(len(item[1])):
                    logger.info('\t    class: {},  probability:  {} \n'.format(
                        self.labels_list[item[1][i]], item[2][i]))
            # save infer results
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


class MulticropMetrics(Metrics):
    def __init__(self, name, mode, metrics_args):
        self.name = name
        self.mode = mode
        if (mode == 'test') or (mode == 'infer'):
            args = {}
            args['num_test_clips'] = metrics_args[mode.upper()][
                'num_test_clips']
            args['dataset_size'] = metrics_args.TEST.dataset_size
            args['filename_gt'] = metrics_args.TEST.filename_gt
            args['checkpoint_dir'] = metrics_args[mode.upper()][
                'checkpoint_dir']
            args['num_classes'] = metrics_args.MODEL.num_classes
            args['labels_list'] = metrics_args.INFER.kinetics_labels
            self.calculator = multicrop_test_metrics.MetricsCalculator(
                name, mode.lower(), **args)
        else:
            self.calculator = kinetics_metrics.MetricsCalculator(name,
                                                                 mode.lower())

    def calculate_and_log_out(self, fetch_list, info=''):
        if (self.mode == 'test') or (self.mode == 'infer'):
            pass
        else:
            if len(fetch_list) == 3:
                loss = fetch_list[0]
                loss = np.mean(np.array(loss))
                pred = fetch_list[1]
                label = fetch_list[2]
            else:
                loss = 0.
                pred = fetch_list[0]
                label = fetch_list[1]
            acc1, acc5 = self.calculator.calculate_metrics(loss, pred, label)
            logger.info(info + '\tLoss: {},\ttop1_acc: {}, \ttop5_acc: {}'.format('%.6f' % loss, \
                                   '%.2f' % acc1, '%.2f' % acc5))

    def accumulate(self, fetch_list):
        if self.mode == 'test':
            pred = fetch_list[0]
            label = fetch_list[1]
            self.calculator.accumulate(pred, label)
        elif self.mode == 'infer':
            pred = fetch_list[0]
            video_id = fetch_list[1]
            self.calculator.accumulate_infer_results(pred, video_id)
        else:
            loss = fetch_list[0]
            pred = fetch_list[1]
            label = fetch_list[2]
            self.calculator.accumulate(loss, pred, label)

    def finalize_and_log_out(self, info='', savedir='./'):
        if self.mode == 'test':
            self.calculator.finalize_metrics()
        elif self.mode == 'infer':
            self.calculator.finalize_infer_metrics()
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


class DetectionMetrics(Metrics):
    def __init__(self, name, mode, cfg):
        self.name = name
        self.mode = mode
        args = {}
        args['score_thresh'] = cfg.TEST.score_thresh
        args['nms_thresh'] = cfg.TEST.nms_thresh
        args['sigma_thresh'] = cfg.TEST.sigma_thresh
        args['soft_thresh'] = cfg.TEST.soft_thresh
        args['class_label_file'] = cfg.TEST.class_label_file
        args['video_duration_file'] = cfg.TEST.video_duration_file
        args['gt_label_file'] = cfg.TEST.filelist
        args['mode'] = mode
        args['name'] = name
        self.calculator = detection_metrics.MetricsCalculator(**args)

    def calculate_and_log_out(self, fetch_list, info=''):
        total_loss = np.array(fetch_list[0])
        loc_loss = np.array(fetch_list[1])
        cls_loss = np.array(fetch_list[2])
        logger.info(
            info + '\tLoss = {}, \tloc_loss = {}, \tcls_loss = {}'.format(
                np.mean(total_loss), np.mean(loc_loss), np.mean(cls_loss)))

    def accumulate(self, fetch_list):
        if self.mode == 'infer':
            self.calculator.accumulate_infer_results(fetch_list)
        else:
            self.calculator.accumulate(fetch_list)

    def finalize_and_log_out(self, info='', savedir='./'):
        if self.mode == 'infer':
            self.calculator.finalize_infer_metrics(savedir)
            #pass
        else:
            self.calculator.finalize_metrics(savedir)
            metrics_dict = self.calculator.get_computed_metrics()
            loss = metrics_dict['avg_loss']
            loc_loss = metrics_dict['avg_loc_loss']
            cls_loss = metrics_dict['avg_cls_loss']
            logger.info(info + '\tLoss: {},\tloc_loss: {}, \tcls_loss: {}'.format('%.6f' % loss, \
                           '%.6f' % loc_loss, '%.6f' % cls_loss))

    def reset(self):
        self.calculator.reset()


class BmnMetrics(Metrics):
    def __init__(self, name, mode, cfg):
        self.name = name
        self.mode = mode
        self.calculator = bmn_proposal_metrics.MetricsCalculator(
            cfg=cfg, name=self.name, mode=self.mode)

    def calculate_and_log_out(self, fetch_list, info=''):
        total_loss = np.array(fetch_list[0])
        tem_loss = np.array(fetch_list[1])
        pem_reg_loss = np.array(fetch_list[2])
        pem_cls_loss = np.array(fetch_list[3])
        logger.info(
            info + '\tLoss = {}, \ttem_loss = {}, \tpem_reg_loss = {}, \tpem_cls_loss = {}'.format(
                '%.04f' % np.mean(total_loss), '%.04f' % np.mean(tem_loss), \
                '%.04f' % np.mean(pem_reg_loss), '%.04f' % np.mean(pem_cls_loss)))

    def accumulate(self, fetch_list):
        if self.mode == 'infer':
            self.calculator.accumulate_infer_results(fetch_list)
        else:
            self.calculator.accumulate(fetch_list)

    def finalize_and_log_out(self, info='', savedir='./'):
        if self.mode == 'infer':
            self.calculator.finalize_infer_metrics()
        else:
            self.calculator.finalize_metrics()
            metrics_dict = self.calculator.get_computed_metrics()
            loss = metrics_dict['avg_loss']
            tem_loss = metrics_dict['avg_tem_loss']
            pem_reg_loss = metrics_dict['avg_pem_reg_loss']
            pem_cls_loss = metrics_dict['avg_pem_cls_loss']
            logger.info(
                info +
                '\tLoss = {}, \ttem_loss = {}, \tpem_reg_loss = {}, \tpem_cls_loss = {}'.
                format('%.04f' % loss, '%.04f' % tem_loss, '%.04f' %
                       pem_reg_loss, '%.04f' % pem_cls_loss))

    def reset(self):
        self.calculator.reset()


class BsnTemMetrics(Metrics):
    def __init__(self, name, mode, cfg):
        self.name = name
        self.mode = mode
        self.calculator = bsn_tem_metrics.MetricsCalculator(
            cfg=cfg, name=self.name, mode=self.mode)

    def calculate_and_log_out(self, fetch_list, info=''):
        total_loss = np.array(fetch_list[0])
        start_loss = np.array(fetch_list[1])
        end_loss = np.array(fetch_list[2])
        action_loss = np.array(fetch_list[3])
        logger.info(
            info +
            '\tLoss = {}, \tstart_loss = {}, \tend_loss = {}, \taction_loss = {}'.
            format('%.04f' % np.mean(total_loss), '%.04f' % np.mean(start_loss),
                   '%.04f' % np.mean(end_loss), '%.04f' % np.mean(action_loss)))

    def accumulate(self, fetch_list):
        if self.mode == 'infer':
            self.calculator.accumulate_infer_results(fetch_list)
        else:
            self.calculator.accumulate(fetch_list)

    def finalize_and_log_out(self, info='', savedir='./'):
        if self.mode == 'infer':
            self.calculator.finalize_infer_metrics()
        else:
            self.calculator.finalize_metrics()
            metrics_dict = self.calculator.get_computed_metrics()
            loss = metrics_dict['avg_loss']
            start_loss = metrics_dict['avg_start_loss']
            end_loss = metrics_dict['avg_end_loss']
            action_loss = metrics_dict['avg_action_loss']
            logger.info(
                info +
                '\tLoss = {}, \tstart_loss = {}, \tend_loss = {}, \taction_loss = {}'.
                format('%.04f' % loss, '%.04f' % start_loss, '%.04f' % end_loss,
                       '%.04f' % action_loss))

    def reset(self):
        self.calculator.reset()


class BsnPemMetrics(Metrics):
    def __init__(self, name, mode, cfg):
        self.name = name
        self.mode = mode
        self.calculator = bsn_pem_metrics.MetricsCalculator(
            cfg=cfg, name=self.name, mode=self.mode)

    def calculate_and_log_out(self, fetch_list, info=''):
        total_loss = np.array(fetch_list[0])
        logger.info(info + '\tLoss = {}'.format('%.04f' % np.mean(total_loss)))

    def accumulate(self, fetch_list):
        if self.mode == 'infer':
            self.calculator.accumulate_infer_results(fetch_list)
        else:
            self.calculator.accumulate(fetch_list)

    def finalize_and_log_out(self, info='', savedir='./'):
        if self.mode == 'infer':
            self.calculator.finalize_infer_metrics()
        else:
            self.calculator.finalize_metrics()
            metrics_dict = self.calculator.get_computed_metrics()
            loss = metrics_dict['avg_loss']
            logger.info(info + '\tLoss = {}'.format('%.04f' % loss))

    def reset(self):
        self.calculator.reset()


class ETSMetrics(Metrics):
    def __init__(self, name, mode, cfg):
        self.name = name
        self.mode = mode
        args = {}
        args['dict_file'] = cfg.MODEL.dict_file
        args['mode'] = mode
        args['name'] = name
        self.calculator = ets_metrics.MetricsCalculator(**args)

    def calculate_and_log_out(self, fetch_list, info=''):
        if (self.mode == 'train') or (self.mode == 'valid'):
            loss = np.array(fetch_list[0])
            logger.info(info + '\tLoss = {}'.format('%.08f' % np.mean(loss)))
        elif self.mode == "test":
            translation_ids = np.array(fetch_list[0])
            translation_scores = np.array(fetch_list[1])
            logger.info(
                info + '\ttranslation_ids = {}, \ttranslation_scores = {}'.
                format('%.01f' % np.mean(translation_ids), '%.04f' % np.mean(
                    translation_scores)))

    def accumulate(self, fetch_list):
        self.calculator.accumulate(fetch_list)

    def finalize_and_log_out(self, info='', savedir='./'):
        if self.mode == 'valid':
            logger.info(info)
        else:  #test or infer
            self.calculator.finalize_metrics(savedir)
            if self.mode == 'test':
                logger.info(
                    info +
                    'please refer to metrics/ets_metrics/README.md to get accuracy'
                )

    def reset(self):
        self.calculator.reset()


class TALLMetrics(Metrics):
    def __init__(self, name, mode, cfg):
        self.name = name
        self.mode = mode
        args = {}
        args['mode'] = mode
        args['name'] = name
        self.calculator = tall_metrics.MetricsCalculator(**args)

    def calculate_and_log_out(self, fetch_list, info=''):
        if (self.mode == 'train') or (self.mode == 'valid'):
            loss = np.array(fetch_list[0])
            logger.info(info + '\tLoss = {}'.format('%.04f' % np.mean(loss)))
        elif self.mode == "test":
            sim_score_mat = np.array(fetch_list[0])
            logger.info(info + '\tsim_score_mat = {}'.format('%.01f' % np.mean(
                sim_score_mat)))

    def accumulate(self, fetch_list):
        self.calculator.accumulate(fetch_list)

    def finalize_and_log_out(self, info='', savedir='./'):
        if self.mode == 'valid':
            logger.info(info)
        elif self.mode == 'infer':
            self.calculator.finalize_infer_metrics(savedir)
        else:
            self.calculator.finalize_metrics(savedir)
            metrics_dict = self.calculator.get_computed_metrics()
            R1_IOU5 = metrics_dict['best_R1_IOU5']
            R5_IOU5 = metrics_dict['best_R5_IOU5']

            logger.info("best_R1_IOU5: {}\n".format(" %0.3f" % R1_IOU5))
            logger.info("best_R5_IOU5: {}\n".format(" %0.3f" % R5_IOU5))

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
regist_metrics("CTCN", DetectionMetrics)
regist_metrics("BMN", BmnMetrics)
regist_metrics("BSNTEM", BsnTemMetrics)
regist_metrics("BSNPEM", BsnPemMetrics)
regist_metrics("ETS", ETSMetrics)
regist_metrics("TALL", TALLMetrics)
