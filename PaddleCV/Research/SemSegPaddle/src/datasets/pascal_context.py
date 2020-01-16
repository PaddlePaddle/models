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


class PascalContextSeg(BaseSeg):
    def __init__(self,
                 file_list,
                 data_dir,
                 shuffle=False,
                 mode=ModelPhase.TRAIN, base_size=520, crop_size=520, rand_scale=True):
        super(PascalContextSeg, self).__init__(file_list, data_dir, shuffle, mode, base_size, crop_size, rand_scale)

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

        if len(img.shape) < 3:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        grt = self._mask_transform(grt)
        return img, grt, img_name, grt_name

