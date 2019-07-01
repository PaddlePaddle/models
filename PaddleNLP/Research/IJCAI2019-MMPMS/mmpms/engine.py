#!/usr/bin/env python
# -*- coding: utf-8 -*-
################################################################################
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
################################################################################

import codecs
import os
import json
import time
import shutil
from collections import defaultdict

from mmpms.utils.logging import getLogger
from mmpms.utils.metrics import Metric, bleu, distinct


def evaluate(model, data_iter):
    metrics_tracker = defaultdict(Metric)
    for batch in data_iter:
        metrics = model.evaluate(inputs=batch)
        for k, v in metrics.items():
            metrics_tracker[k].update(v, batch["size"])
    return metrics_tracker


def flatten_batch(batch):
    examples = []
    for vs in zip(*batch.values()):
        ex = dict(zip(batch.keys(), vs))
        examples.append(ex)
    return examples


def infer(model, data_iter, parse_dict, save_file=None):
    results = []
    for batch in data_iter:
        result = model.infer(inputs=batch)
        batch_result = {}

        # denumericalization
        for k, parse_fn in parse_dict.items():
            if k in result:
                batch_result[k] = parse_fn(result[k])

        results += flatten_batch(batch_result)

    if save_file is not None:
        with codecs.open(save_file, "w", encoding="utf-8") as fp:
            json.dump(results, fp, ensure_ascii=False, indent=2)
        print("Saved inference results to '{}'".format(save_file))
    return results


class Engine(object):
    def __init__(self,
                 model,
                 valid_metric_name="-loss",
                 num_epochs=1,
                 save_dir=None,
                 log_steps=None,
                 valid_steps=None,
                 logger=None):
        self.model = model

        self.is_decreased_valid_metric = valid_metric_name[0] == "-"
        self.valid_metric_name = valid_metric_name[1:]
        self.num_epochs = num_epochs
        self.save_dir = save_dir or "./"
        self.log_steps = log_steps
        self.valid_steps = valid_steps

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.logger = logger or logging.getLogger(
            os.path.join(self.save_dir, "run.log"))

        best_valid_metric = float("inf") if self.is_decreased_valid_metric \
            else -float("inf")
        self.state = {
            "epoch": 0,
            "iteration": 0,
            "best_valid_metric": best_valid_metric
        }

    @property
    def epoch(self):
        return self.state["epoch"]

    @property
    def iteration(self):
        return self.state["iteration"]

    @property
    def best_valid_metric(self):
        return self.state["best_valid_metric"]

    def train_epoch(self, train_iter, valid_iter=None):
        self.state["epoch"] += 1
        num_batches = len(train_iter)
        metrics_tracker = defaultdict(Metric)
        for batch_id, batch in enumerate(train_iter, 1):

            # Do a training iteration
            start_time = time.time()
            metrics = self.model.train(inputs=batch)
            elapsed = time.time() - start_time

            for k, v in metrics.items():
                metrics_tracker[k].update(v, batch["size"])
            metrics_tracker["time"].update(elapsed)
            self.state["iteration"] += 1

            if self.log_steps and batch_id % self.log_steps == 0:
                metrics_message = [
                    "{}-{}".format(name.upper(), metric.val)
                    for name, metric in metrics_tracker.items()
                ]
                message_prefix = "[Train][{}][{}/{}]".format(
                    self.epoch, batch_id, num_batches)
                message = "   ".join([message_prefix] + metrics_message)
                self.logger.info(message)

            if self.valid_steps and valid_iter is not None and \
                    batch_id % self.valid_steps == 0:
                self.evaluate(valid_iter)

        if valid_iter is not None:
            self.evaluate(valid_iter)

    def save(self, is_best):
        model_file = os.path.join(self.save_dir,
                                  "model_epoch_{}".format(self.epoch))
        self.model.save(model_file)
        self.logger.info("Saved model to '{}'".format(model_file))

        if is_best:
            best_model_file = os.path.join(self.save_dir, "best_model")
            if os.path.isdir(model_file):
                if os.path.exists(best_model_file):
                    shutil.rmtree(best_model_file)
                shutil.copytree(model_file, best_model_file)
            else:
                shutil.copyfile(model_file, best_model_file)
            self.logger.info("Saved best model to '{}' "
                             "with new best valid metric "
                             "{}-{}".format(best_model_file,
                                            self.valid_metric_name.upper(),
                                            self.best_valid_metric))

    def load(self, model_dir):
        self.model.load(model_dir)
        self.logger.info("Loaded model checkpoint from {}".format(model_dir))

    def evaluate(self, data_iter, is_save=True):
        metrics_tracker = evaluate(self.model, data_iter)
        metrics_message = [
            "{}-{}".format(name.upper(), metric.avg)
            for name, metric in metrics_tracker.items()
        ]
        message_prefix = "[Valid][{}]".format(self.epoch)
        message = "   ".join([message_prefix] + metrics_message)
        self.logger.info(message)

        if is_save:
            cur_valid_metric = metrics_tracker.get(self.valid_metric_name).avg
            if self.is_decreased_valid_metric:
                is_best = cur_valid_metric < self.best_valid_metric
            else:
                is_best = cur_valid_metric > self.best_valid_metric
            if is_best:
                self.state["best_valid_metric"] = cur_valid_metric
            self.save(is_best)
