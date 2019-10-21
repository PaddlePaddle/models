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
"""
Trainer class.
"""

import json
import logging
import os
import sys
import time

import numpy as np
import paddle.fluid as fluid
import paddle.fluid.dygraph as dygraph
from tqdm import tqdm

from args import str2bool
from dataloader import DataLoader
from metrics.metrics_tracker import MetricsTracker
from metrics.metrics import bleu
from metrics.metrics import distinct
import modules.parallel as parallel


def get_logger(log_path, name="default"):
    logger = logging.getLogger(name)
    logger.propagate = False
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(message)s")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


def evaluate_generation_result(results):
    tgt = [result["tgt"].split(" ") for result in results]
    pred = [result["preds"][np.argmax(result["scores"])] for result in results]
    pred = [p.split(" ") for p in pred]
    metrics = {}
    metrics_tracker = MetricsTracker()

    bleu1, bleu2 = bleu(pred, tgt)
    metrics.update({"bleu_1": bleu1, "bleu_2": bleu2})

    intra_dist1, intra_dist2, inter_dist1, inter_dist2 = distinct(pred)
    metrics.update({"intra_dist_1": intra_dist1,
                    "intra_dist_2": intra_dist2,
                    "inter_dist_1": inter_dist1,
                    "inter_dist_2": inter_dist2})

    avg_len = sum(map(len, pred)) / len(pred)
    metrics.update({"len": avg_len})

    metrics_tracker.update(metrics, num_samples=1)
    return metrics_tracker


def save(model, model_path):
    if isinstance(model, parallel.DataParallel):
        model = model._layers
    dygraph.save_persistables(model.state_dict(), model_path, optimizers=model.optimizer)
    return


class Trainer(object):

    @classmethod
    def add_cmdline_argument(cls, parser):
        """ Add the cmdline arguments of trainer. """
        group = parser.add_argument_group("Trainer")
        group.add_argument("--use_data_distributed", type=str2bool, default=False,
                           help="Whether to use data distributed for parallel training.")
        group.add_argument("--valid_metric_name", type=str, default="-loss",
                           help="The validation metric determining which checkpoint is the best.")
        group.add_argument("--num_epochs", type=int, default=10,
                           help="Total number of training epochs to perform.")
        group.add_argument("--save_dir", type=str, required=True,
                           help="The output directory where the model will be saved.")
        group.add_argument("--batch_size", type=int, default=8,
                           help="Total batch size for training/evaluation/inference.")
        group.add_argument("--log_steps", type=int, default=100,
                           help="The number of training steps to output current metrics "
                           "on past training dataset.")
        group.add_argument("--valid_steps", type=int, default=2000,
                           help="The number of training steps to perform a evaluation "
                           "on validation datasets.")
        group.add_argument("--save_checkpoint", type=str2bool, default=True,
                           help="Whether to save one checkpoints for each training epoch.")
        group.add_argument("--save_summary", type=str2bool, default=False,
                           help="Whether to save metrics summary for visualDL module.")
        DataLoader.add_cmdline_argument(group)
        return group

    def __init__(self, model, to_tensor, hparams, logger=None):
        # Use data distributed
        if hparams.use_data_distributed:
            strategy = parallel.prepare_context()
            parallel_model = parallel.DataParallel(model, strategy)
            model.before_backward_fn = parallel_model.scale_loss
            model.after_backward_fn = parallel_model.apply_collective_grads
            model = parallel_model

        self.model = model
        self.to_tensor = to_tensor

        self.is_decreased_valid_metric = hparams.valid_metric_name[0] == "-"
        self.valid_metric_name = hparams.valid_metric_name[1:]
        self.num_epochs = hparams.num_epochs
        self.save_dir = hparams.save_dir
        self.log_steps = hparams.log_steps
        self.valid_steps = hparams.valid_steps
        self.save_checkpoint = hparams.save_checkpoint
        self.save_summary = hparams.save_summary

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.logger = logger or get_logger(os.path.join(self.save_dir, "trainer.log"), "trainer")

        if self.save_summary:
            from visualdl import LogWriter
            self.summary_logger = LogWriter(os.path.join(self.save_dir, "summary"), sync_cycle=10000)
            self.train_summary = {}
            self.valid_summary = {}

        self.metrics_tracker = MetricsTracker()

        self.best_valid_metric = float("inf" if self.is_decreased_valid_metric else "-inf")
        self.epoch = 0
        self.batch_num = 0

    def train_epoch(self, train_iter, valid_iter, infer_iter=None, infer_parse_dict=None):
        """
        Train an epoch.

        @param train_iter
        @type : DataLoader

        @param valid_iter
        @type : DataLoader

        @param infer_iter
        @type : DataLoader

        @param infer_parse_dict
        @type : dict of function
        """
        self.epoch += 1
        num_batches = len(train_iter)
        self.metrics_tracker.clear()
        times = []
        for batch_id, (batch, batch_size) in enumerate(train_iter, 1):
            batch = type(batch)(map(lambda kv: (kv[0], self.to_tensor(kv[1])), batch.items()))
            batch["epoch"] = self.epoch
            batch["num_steps"] = self.batch_num
            # measure data loading time

            # Do a training iteration
            start_time = time.time()
            metrics = self.model(batch, is_training=True)
            elapsed = time.time() - start_time
            times.append(elapsed)

            self.metrics_tracker.update(metrics, batch_size)
            self.batch_num += 1

            if self.log_steps and batch_id % self.log_steps == 0:
                metrics_message = self.metrics_tracker.value()
                message_prefix = f"[Train][{self.epoch}][{batch_id}/{num_batches}]"
                avg_time = f"AVG_Time-{sum(times[-self.log_steps:]) / self.log_steps:.3f}"
                message = "   ".join([message_prefix, metrics_message, avg_time])
                self.logger.info(message)

            if self.save_summary:
                with self.summary_logger.mode("train"):
                    for k, v in self.metrics_tracker.items():
                        if k not in self.train_summary:
                            self.train_summary[k] = self.summary_logger.scalar(k)
                        scalar = self.train_summary[k]
                        scalar.add_record(self.batch_num, v)

            if self.valid_steps and valid_iter is not None and \
                    batch_id % self.valid_steps == 0:
                self.evaluate(valid_iter)

        if valid_iter is not None:
            self.evaluate(valid_iter)

        if infer_iter is not None and infer_parse_dict is not None:
            self.infer(infer_iter, infer_parse_dict)

        return

    def infer(self, data_iter, parse_dict, num_batches=None):
        """
        Inference interface.

        @param : data_iter
        @type : DataLoader

        @param : parse_dict
        @type : dict of function

        @param : num_batches : the number of batch to infer
        @type : int/None
        """
        self.logger.info("Generation starts ...")
        infer_save_file = os.path.join(self.save_dir, f"infer_{self.epoch}.result.json")
        # Inference
        infer_results = []
        batch_cnt = 0
        for batch, batch_size in tqdm(data_iter, total=num_batches):
            batch = type(batch)(map(lambda kv: (kv[0], self.to_tensor(kv[1])), batch.items()))

            result = self.model.infer(inputs=batch)
            batch_result = {}

            def to_list(batch):
                """ Parse list. """
                return batch.tolist()

            # parse
            for k in result:
                if k in parse_dict:
                    parse_fn = parse_dict[k]
                else:
                    parse_fn = to_list
                if result[k] is not None:
                    batch_result[k] = parse_fn(result[k])

            for vs in zip(*batch_result.values()):
                infer_result = {}
                for k, v in zip(batch_result.keys(), vs):
                    infer_result[k] = v
                infer_results.append(infer_result)

            batch_cnt += 1
            if batch_cnt == num_batches:
                break

        self.logger.info(f"Saved inference results to {infer_save_file}")
        with open(infer_save_file, "w") as fp:
            json.dump(infer_results, fp, indent=2)
        infer_metrics_tracker = evaluate_generation_result(infer_results)
        metrics_message = infer_metrics_tracker.summary()
        message_prefix = f"[Infer][{self.epoch}]"
        message = "   ".join([message_prefix, metrics_message])
        self.logger.info(message)
        return

    def evaluate(self, data_iter, need_save=True):
        """
        Evaluation interface

        @param : data_iter
        @type : DataLoader

        @param : need_save
        @type : bool
        """
        if isinstance(self.model, parallel.DataParallel):
            need_save = need_save and parallel.Env().local_rank == 0

        # Evaluation
        metrics_tracker = MetricsTracker()
        for batch, batch_size in data_iter:
            batch = type(batch)(map(lambda kv: (kv[0], self.to_tensor(kv[1])), batch.items()))
            metrics = self.model(batch, is_training=False)
            metrics_tracker.update(metrics, batch_size)
        metrics_message = metrics_tracker.summary()
        message_prefix = f"[Valid][{self.epoch}]"
        message = "   ".join([message_prefix, metrics_message])
        self.logger.info(message)

        # Check valid metric
        cur_valid_metric = metrics_tracker.get(self.valid_metric_name)
        if self.is_decreased_valid_metric:
            is_best = cur_valid_metric < self.best_valid_metric
        else:
            is_best = cur_valid_metric > self.best_valid_metric
        if is_best and need_save:
            # Save current best model
            self.best_valid_metric = cur_valid_metric
            best_model_path = os.path.join(self.save_dir, "best.model")
            save(self.model, best_model_path)
            self.logger.info(
                f"Saved best model to '{best_model_path}' with new best valid metric "
                f"{self.valid_metric_name.upper()}-{self.best_valid_metric:.3f}")
        
        # Save checkpoint
        if self.save_checkpoint and need_save:
            model_file = os.path.join(self.save_dir, f"epoch_{self.epoch}.model")
            save(self.model, model_file)

        if self.save_summary and need_save:
            with self.summary_logger.mode("valid"):
                for k, v in self.metrics_tracker.items():
                    if k not in self.valid_summary:
                        self.valid_summary[k] = self.summary_logger.scalar(k)
                    scalar = self.valid_summary[k]
                    scalar.add_record(self.batch_num, v)

        return
