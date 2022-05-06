# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import copy
import time

import numpy as np
from PIL import Image, ImageOps, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import paddle
import paddle.distributed as dist
from paddle.distributed import fleet
from paddle import amp
from paddle.static import InputSpec

from lib.utils.workspace import create
from lib.utils.checkpoint import load_weight, load_pretrain_weight
from lib.utils.visualizer import visualize_results, save_result
from lib.metrics.coco_utils import get_infer_results
from lib.metrics import KeyPointTopDownCOCOEval
from lib.dataset.category import get_categories
import lib.utils.stats as stats

from .callbacks import Callback, ComposeCallback, LogPrinter, Checkpointer, VisualDLWriter
from .export_utils import _dump_infer_config, _prune_input_spec

from lib.utils.logger import setup_logger
logger = setup_logger('hrnet.pose')

__all__ = ['Trainer']


class Trainer(object):
    def __init__(self, cfg, mode='train'):
        self.cfg = cfg
        assert mode.lower() in ['train', 'eval', 'test'], \
                "mode should be 'train', 'eval' or 'test'"
        self.mode = mode.lower()
        self.optimizer = None

        # init distillation config
        self.distill_model = None
        self.distill_loss = None

        # build data loader
        self.dataset = cfg['{}Dataset'.format(self.mode.capitalize())]

        if self.mode == 'train':
            self.loader = create('{}Reader'.format(self.mode.capitalize()))(
                self.dataset, cfg.worker_num)

        self.model = create(cfg.architecture)

        #normalize params for deploy
        self.model.load_meanstd(cfg['TestReader']['sample_transforms'])

        # EvalDataset build with BatchSampler to evaluate in single device
        if self.mode == 'eval':
            self._eval_batch_sampler = paddle.io.BatchSampler(
                self.dataset, batch_size=self.cfg.EvalReader['batch_size'])
            self.loader = create('{}Reader'.format(self.mode.capitalize()))(
                self.dataset, cfg.worker_num, self._eval_batch_sampler)
        # TestDataset build after user set images, skip loader creation here

        self._nranks = dist.get_world_size()
        self._local_rank = dist.get_rank()

        self.status = {}

        self.start_epoch = 0
        self.end_epoch = 0 if 'epoch' not in cfg else cfg.epoch

        # initial default callbacks
        self._init_callbacks()

        # initial default metrics
        self._init_metrics()
        self._reset_metrics()

    def _init_callbacks(self):
        if self.mode == 'train':
            self._callbacks = [LogPrinter(self), Checkpointer(self)]
            if self.cfg.get('use_vdl', False):
                self._callbacks.append(VisualDLWriter(self))
            self._compose_callback = ComposeCallback(self._callbacks)
        elif self.mode == 'eval':
            self._callbacks = [LogPrinter(self)]
            self._compose_callback = ComposeCallback(self._callbacks)
        elif self.mode == 'test' and self.cfg.get('use_vdl', False):
            self._callbacks = [VisualDLWriter(self)]
            self._compose_callback = ComposeCallback(self._callbacks)
        else:
            self._callbacks = []
            self._compose_callback = None

    def _init_metrics(self, validate=False):
        if self.mode == 'test' or (self.mode == 'train' and not validate):
            self._metrics = []
            return
        if self.cfg.metric == 'KeyPointTopDownCOCOEval':
            eval_dataset = self.cfg['EvalDataset']
            eval_dataset.check_or_download_dataset()
            anno_file = eval_dataset.get_anno()
            save_prediction_only = self.cfg.get('save_prediction_only', False)
            self._metrics = [
                KeyPointTopDownCOCOEval(
                    anno_file,
                    len(eval_dataset),
                    self.cfg.num_joints,
                    self.cfg.save_dir,
                    save_prediction_only=save_prediction_only)
            ]
        else:
            logger.warning("Metric not support for metric type {}".format(
                self.cfg.metric))
            self._metrics = []

    def init_optimizer(self, ):
        # build optimizer in train mode
        if self.mode == 'train':
            steps_per_epoch = len(self.loader)
            self.lr = create('LearningRate')(steps_per_epoch)
            self.optimizer = create('OptimizerBuilder')(
                self.lr, [self.model, self.distill_model])

    def _reset_metrics(self):
        for metric in self._metrics:
            metric.reset()

    def register_callbacks(self, callbacks):
        callbacks = [c for c in list(callbacks) if c is not None]
        for c in callbacks:
            assert isinstance(c, Callback), \
                    "metrics shoule be instances of subclass of Metric"
        self._callbacks.extend(callbacks)
        self._compose_callback = ComposeCallback(self._callbacks)

    def register_metrics(self, metrics):
        metrics = [m for m in list(metrics) if m is not None]
        self._metrics.extend(metrics)

    def load_weights(self, weights, model=None):
        self.start_epoch = 0
        if model is None:
            model = self.model
        load_pretrain_weight(self.model, weights)
        logger.debug("Load weights {} to start training".format(weights))

    def train(self, validate=False):
        assert self.mode == 'train', "Model not in 'train' mode"
        Init_mark = False

        model = self.model
        if self._nranks > 1:
            model = paddle.DataParallel(
                self.model,
                find_unused_parameters=self.cfg.get("find_unused_parameters",
                                                    False))

        self.status.update({
            'epoch_id': self.start_epoch,
            'step_id': 0,
            'steps_per_epoch': len(self.loader)
        })

        self.status['batch_time'] = stats.SmoothedValue(
            self.cfg.log_iter, fmt='{avg:.4f}')
        self.status['data_time'] = stats.SmoothedValue(
            self.cfg.log_iter, fmt='{avg:.4f}')
        self.status['training_staus'] = stats.TrainingStats(self.cfg.log_iter)

        self._compose_callback.on_train_begin(self.status)

        for epoch_id in range(self.start_epoch, self.cfg.epoch):
            self.status['mode'] = 'train'
            self.status['epoch_id'] = epoch_id
            self._compose_callback.on_epoch_begin(self.status)
            self.loader.dataset.set_epoch(epoch_id)
            model.train()
            iter_tic = time.time()
            for step_id, data in enumerate(self.loader):
                self.status['data_time'].update(time.time() - iter_tic)
                self.status['step_id'] = step_id
                self._compose_callback.on_step_begin(self.status)
                data['epoch_id'] = epoch_id

                # model forward
                outputs = model(data)
                if self.distill_model is not None:
                    teacher_outputs = self.distill_model(data)
                    distill_loss = self.distill_loss(outputs, teacher_outputs,
                                                     data)
                    loss = outputs['loss'] + teacher_outputs[
                        "loss"] + distill_loss
                else:
                    loss = outputs['loss']
                # model backward
                loss.backward()
                self.optimizer.step()
                curr_lr = self.optimizer.get_lr()
                self.lr.step()
                self.optimizer.clear_grad()
                self.status['learning_rate'] = curr_lr

                if self._nranks < 2 or self._local_rank == 0:
                    loss_dict = {"loss": outputs['loss']}
                    if self.distill_model is not None:
                        loss_dict.update({
                            "loss_student": outputs['loss'],
                            "loss_teacher": teacher_outputs["loss"],
                            "loss_distill": distill_loss,
                            "loss": loss
                        })
                    self.status['training_staus'].update(loss_dict)

                self.status['batch_time'].update(time.time() - iter_tic)
                self._compose_callback.on_step_end(self.status)
                iter_tic = time.time()

            self._compose_callback.on_epoch_end(self.status)

            if validate and self._local_rank == 0 \
                    and ((epoch_id + 1) % self.cfg.snapshot_epoch == 0 \
                             or epoch_id == self.end_epoch - 1):
                print("begin to eval...")
                if not hasattr(self, '_eval_loader'):
                    # build evaluation dataset and loader
                    self._eval_dataset = self.cfg.EvalDataset
                    self._eval_batch_sampler = \
                        paddle.io.BatchSampler(
                            self._eval_dataset,
                            batch_size=self.cfg.EvalReader['batch_size'])
                    self._eval_loader = create('EvalReader')(
                        self._eval_dataset,
                        self.cfg.worker_num,
                        batch_sampler=self._eval_batch_sampler)
                # if validation in training is enabled, metrics should be re-init
                # Init_mark makes sure this code will only execute once
                if validate and Init_mark == False:
                    Init_mark = True
                    self._init_metrics(validate=validate)
                    self._reset_metrics()
                with paddle.no_grad():
                    self.status['save_best_model'] = True
                    self._eval_with_loader(self._eval_loader)

        self._compose_callback.on_train_end(self.status)

    def _eval_with_loader(self, loader):
        sample_num = 0
        tic = time.time()
        self._compose_callback.on_epoch_begin(self.status)
        self.status['mode'] = 'eval'
        self.model.eval()
        for step_id, data in enumerate(loader):
            self.status['step_id'] = step_id
            self._compose_callback.on_step_begin(self.status)
            # forward
            outs = self.model(data)

            # update metrics
            for metric in self._metrics:
                metric.update(data, outs)

            sample_num += data['im_id'].numpy().shape[0]
            self._compose_callback.on_step_end(self.status)

        self.status['sample_num'] = sample_num
        self.status['cost_time'] = time.time() - tic

        # accumulate metric to log out
        for metric in self._metrics:
            metric.accumulate()
            metric.log()
        self._compose_callback.on_epoch_end(self.status)
        # reset metric states for metric may performed multiple times
        self._reset_metrics()

    def evaluate(self):
        with paddle.no_grad():
            self._eval_with_loader(self.loader)

    def predict(self,
                images,
                draw_threshold=0.5,
                output_dir='output',
                save_txt=False):
        self.dataset.set_images(images)
        loader = create('TestReader')(self.dataset, 0)

        imid2path = self.dataset.get_imid2path()

        anno_file = self.dataset.get_anno()
        clsid2catid, catid2name = get_categories(
            self.cfg.metric, anno_file=anno_file)

        # Run Infer 
        self.status['mode'] = 'test'
        self.model.eval()
        results = []
        for step_id, data in enumerate(loader):
            self.status['step_id'] = step_id
            # forward
            outs = self.model(data)

            for key in ['im_shape', 'scale_factor', 'im_id']:
                outs[key] = data[key]
            for key, value in outs.items():
                if hasattr(value, 'numpy'):
                    outs[key] = value.numpy()
            results.append(outs)

        for outs in results:
            batch_res = get_infer_results(outs, clsid2catid)
            bbox_num = outs['bbox_num']

            start = 0
            for i, im_id in enumerate(outs['im_id']):
                image_path = imid2path[int(im_id)]
                image = Image.open(image_path).convert('RGB')
                image = ImageOps.exif_transpose(image)
                self.status['original_image'] = np.array(image.copy())

                end = start + bbox_num[i]
                bbox_res = batch_res['bbox'][start:end] \
                        if 'bbox' in batch_res else None
                keypoint_res = batch_res['keypoint'][start:end] \
                        if 'keypoint' in batch_res else None
                image = visualize_results(image, bbox_res, keypoint_res,
                                          int(im_id), catid2name,
                                          draw_threshold)
                self.status['result_image'] = np.array(image.copy())
                if self._compose_callback:
                    self._compose_callback.on_step_end(self.status)
                # save image with detection
                save_name = self._get_save_image_name(output_dir, image_path)
                logger.info("Detection bbox results save in {}".format(
                    save_name))
                image.save(save_name, quality=95)
                if save_txt:
                    save_path = os.path.splitext(save_name)[0] + '.txt'
                    results = {}
                    results["im_id"] = im_id
                    if bbox_res:
                        results["bbox_res"] = bbox_res
                    if keypoint_res:
                        results["keypoint_res"] = keypoint_res
                    save_result(save_path, results, catid2name, draw_threshold)
                start = end

    def _get_save_image_name(self, output_dir, image_path):
        """
        Get save image name from source image path.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        image_name = os.path.split(image_path)[-1]
        name, ext = os.path.splitext(image_name)
        return os.path.join(output_dir, "{}".format(name)) + ext

    def _get_infer_cfg_and_input_spec(self, save_dir, prune_input=True):
        image_shape = [3, -1, -1]
        im_shape = [None, 2]
        scale_factor = [None, 2]
        test_reader_name = 'TestReader'
        if 'inputs_def' in self.cfg[test_reader_name]:
            inputs_def = self.cfg[test_reader_name]['inputs_def']
            image_shape = inputs_def.get('image_shape', None)
        # set image_shape=[None, 3, -1, -1] as default

        image_shape = [None] + image_shape

        if hasattr(self.model, 'deploy'):
            self.model.deploy = True

        # Save infer cfg
        _dump_infer_config(self.cfg,
                           os.path.join(save_dir, 'infer_cfg.yml'),
                           image_shape, self.model)

        input_spec = [{
            "image": InputSpec(
                shape=image_shape, name='image'),
            "im_shape": InputSpec(
                shape=im_shape, name='im_shape'),
            "scale_factor": InputSpec(
                shape=scale_factor, name='scale_factor')
        }]

        if prune_input:
            static_model = paddle.jit.to_static(
                self.model, input_spec=input_spec)
            # NOTE: dy2st do not pruned program, but jit.save will prune program
            # input spec, prune input spec here and save with pruned input spec
            pruned_input_spec = _prune_input_spec(
                input_spec, static_model.forward.main_program,
                static_model.forward.outputs)
        else:
            static_model = None
            pruned_input_spec = input_spec

        return static_model, pruned_input_spec

    def export(self, output_dir='output_inference'):
        self.model.eval()
        model_name = os.path.splitext(os.path.split(self.cfg.filename)[-1])[0]
        save_dir = os.path.join(output_dir, model_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        static_model, pruned_input_spec = self._get_infer_cfg_and_input_spec(
            save_dir)

        # save model
        if 'slim' not in self.cfg:
            paddle.jit.save(
                static_model,
                os.path.join(save_dir, 'model'),
                input_spec=pruned_input_spec)
        else:
            self.cfg.slim.save_quantized_model(
                self.model,
                os.path.join(save_dir, 'model'),
                input_spec=pruned_input_spec)
        logger.info("Export model and saved in {}".format(save_dir))
