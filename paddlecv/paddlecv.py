# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import os
import sys
import importlib
import argparse

__dir__ = os.path.dirname(__file__)

sys.path.insert(0, os.path.join(__dir__, ''))

import cv2
import logging
import numpy as np
from pathlib import Path

ppcv = importlib.import_module('.', 'ppcv')
tools = importlib.import_module('.', 'tools')
tests = importlib.import_module('.', 'tests')

VERSION = '0.1.0'

import yaml
from ppcv.model_zoo.model_zoo import TASK_DICT, list_model, get_config_file
from ppcv.engine.pipeline import Pipeline
from ppcv.utils.logger import setup_logger

logger = setup_logger()


class PaddleCV(object):
    def __init__(self,
                 task_name=None,
                 config_path=None,
                 output_dir='output',
                 run_mode='paddle',
                 device='CPU'):

        if task_name is not None:
            assert task_name in TASK_DICT, f"task_name must be one of {list(TASK_DICT.keys())} but got {task_name}"
            config_path = get_config_file(task_name)
        else:
            assert config_path is not None, "task_name and config_path can not be None at the same time!!!"

        self.cfg_dict = dict(
            config=config_path,
            output_dir=output_dir,
            run_mode=run_mode,
            device=device)
        cfg = argparse.Namespace(**self.cfg_dict)
        self.pipeline = Pipeline(cfg)

    @classmethod
    def list_all_supported_tasks(self, ):
        logger.info(
            f"Tasks and recommanded configs that paddlecv supports are : ")
        buffer = yaml.dump(TASK_DICT)
        print(buffer)
        return

    @classmethod
    def list_all_supported_models(self, filters=[]):
        list_model(filters)
        return

    def __call__(self, input):
        res = self.pipeline.run(input)
        return res
