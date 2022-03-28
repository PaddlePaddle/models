# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

# add python path of PadleDetection to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
sys.path.insert(0, parent_path)

# ignore warning log
import warnings
warnings.filterwarnings('ignore')
import copy

import paddle

from lib.utils.workspace import load_config
from lib.utils.env import init_parallel_env, set_random_seed, init_fleet_env
from lib.utils.logger import setup_logger
from lib.utils.checkpoint import load_pretrain_weight
from lib.utils.cli import ArgsParser
from lib.utils.check import check_config, check_gpu, check_version
from lib.core.trainer import Trainer
from lib.utils.workspace import create
from lib.models.loss import DistMSELoss
from lib.slim import build_slim_model

logger = setup_logger('train')


def build_teacher_model(config):
    model = create(config.architecture)
    if config.get('pretrain_weights', None):
        load_pretrain_weight(model, config.pretrain_weights)
        logger.debug("Load weights {} to start training".format(
            config.pretrain_weights))
    if config.get('weights', None):
        load_pretrain_weight(model, config.weights)
        logger.debug("Load weights {} to start training".format(
            config.weights))

    if config.get("freeze_parameters", True):
        for param in model.parameters():
            param.trainable = False

    model.train()
    return model


def build_distill_loss(config):
    loss_config = copy.deepcopy(config["distill_loss"])
    name = loss_config.pop("name")
    dist_loss_class = eval(name)(**loss_config)
    return dist_loss_class


def parse_args():
    parser = ArgsParser()
    parser.add_argument(
        "--eval",
        action='store_true',
        default=False,
        help="Whether to perform evaluation in train")
    parser.add_argument(
        "--distill_config",
        default=None,
        type=str,
        help="Configuration file of model distillation.")
    parser.add_argument(
        "--enable_ce",
        type=bool,
        default=False,
        help="If set True, enable continuous evaluation job."
        "This flag is only used for internal test.")
    parser.add_argument(
        "--use_vdl",
        type=bool,
        default=False,
        help="whether to record the data to VisualDL.")
    parser.add_argument(
        '--vdl_log_dir',
        type=str,
        default="vdl_log_dir/scalar",
        help='VisualDL logging directory for scalar.')

    args = parser.parse_args()
    return args


def run(FLAGS, cfg):
    init_parallel_env()

    if FLAGS.enable_ce:
        set_random_seed(0)

    # build trainer
    trainer = Trainer(cfg, mode='train')

    # load weights
    if 'pretrain_weights' in cfg and cfg.pretrain_weights:
        trainer.load_weights(cfg.pretrain_weights)

    # init config
    if FLAGS.distill_config is not None:
        distill_config = load_config(FLAGS.distill_config)
        trainer.distill_model = build_teacher_model(distill_config)
        trainer.distill_loss = build_distill_loss(distill_config)

    trainer.init_optimizer()

    # training
    trainer.train(FLAGS.eval)


def main():
    FLAGS = parse_args()
    cfg = load_config(FLAGS.config)
    cfg['eval'] = FLAGS.eval
    cfg['enable_ce'] = FLAGS.enable_ce
    cfg['distill_config'] = FLAGS.distill_config
    cfg['use_vdl'] = FLAGS.use_vdl
    cfg['vdl_log_dir'] = FLAGS.vdl_log_dir
    cfg['distill_config'] = FLAGS.distill_config

    if cfg.use_gpu:
        paddle.set_device('gpu')
    else:
        paddle.set_device('cpu')

    if 'slim' in cfg:
        cfg = build_slim_model(cfg)

    check_config(cfg)
    check_gpu(cfg.use_gpu)
    check_version()

    run(FLAGS, cfg)


if __name__ == "__main__":
    main()
