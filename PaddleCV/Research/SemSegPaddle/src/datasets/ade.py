# coding: utf8
# copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function
import sys
import os
import math
import random
import functools
import io
import time
import codecs
import numpy as np
import paddle
import paddle.fluid as fluid
import cv2
from PIL import Image
import copy

from src.utils.config import cfg
from src.models.model_builder import ModelPhase
from .baseseg import BaseSeg


class AdeSeg(BaseSeg):
    def __init__(self,
                 file_list,
                 data_dir,
                 shuffle=False,
                 mode=ModelPhase.TRAIN, base_size=520, crop_size=520, rand_scale=True):
        super(AdeSeg, self).__init__(file_list, data_dir, shuffle, mode, base_size, crop_size, rand_scale)

    def _mask_transform(self, mask):
        target = np.array(mask).astype('int32') - 1
        return target



    def load_image(self, line, src_dir, mode=ModelPhase.TRAIN):
        # original image cv2.imread flag setting
        cv2_imread_flag = cv2.IMREAD_COLOR
        if cfg.DATASET.IMAGE_TYPE == "rgba":
            # If use RBGA 4 channel ImageType, use IMREAD_UNCHANGED flags to
            # reserver alpha channel
            cv2_imread_flag = cv2.IMREAD_UNCHANGED
        #print("line: ", line)
        parts = line.strip().split(cfg.DATASET.SEPARATOR)
        if len(parts) != 2:
            if mode == ModelPhase.TRAIN or mode == ModelPhase.EVAL:
                raise Exception("File list format incorrect! It should be"
                                " image_name{}label_name\\n".format(
                                    cfg.DATASET.SEPARATOR))
            img_name, grt_name = parts[0], None
        else:
            img_name, grt_name = parts[0], parts[1]

        img_path = os.path.join(src_dir, img_name)
        img = self.cv2_imread(img_path, cv2_imread_flag)

        if grt_name is not None:
            grt_path = os.path.join(src_dir, grt_name)
            grt = self.pil_imread(grt_path)
        else:
            grt = None

        if img is None:
            raise Exception(
                "Empty image, src_dir: {}, img: {} & lab: {}".format(
                    src_dir, img_path, grt_path))

        img_height = img.shape[0]
        img_width = img.shape[1]
        #print('img.shape',img.shape)
        if grt is not None:
            grt_height = grt.shape[0]
            grt_width = grt.shape[1]

            if img_height != grt_height or img_width != grt_width:
                raise Exception(
                    "source img and label img must has the same size")
        else:
            if mode == ModelPhase.TRAIN or mode == ModelPhase.EVAL:
                raise Exception(
                    "Empty image, src_dir: {}, img: {} & lab: {}".format(
                        src_dir, img_path, grt_path))

        if len(img.shape) < 3:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        grt = self._mask_transform(grt)

        return img, grt, img_name, grt_name

