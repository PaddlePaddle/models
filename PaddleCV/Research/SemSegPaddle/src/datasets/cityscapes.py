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


class CityscapesSeg(BaseSeg):
    def __init__(self, file_list, data_dir, shuffle=False, mode=ModelPhase.TRAIN, base_size=1024, crop_size=769, rand_scale=True):

        super(CityscapesSeg, self).__init__(file_list, data_dir, shuffle, mode, base_size, crop_size, rand_scale)

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
                raise Exception("File list format incorrect! It should be image_name {} label_name\\n".format(cfg.DATASET.SEPARATOR))
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

        img_height = img.shape[0]
        img_width = img.shape[1]
        if grt is not None:
            grt_height = grt.shape[0]
            grt_width = grt.shape[1]
            id_to_trainid = [255, 255, 255, 255, 255,
                             255, 255, 255, 0, 1,
                             255, 255, 2, 3, 4,
                             255, 255, 255, 5, 255,
                             6, 7, 8, 9, 10,
                             11, 12, 13, 14, 15,
                             255, 255, 16, 17, 18]
            grt_ = np.zeros([grt_height, grt_width])
        
            for h in range(grt_height):
                for w in range(grt_width):
                    grt_[h][w] = id_to_trainid[int(grt[h][w])+1]

            if img_height != grt_height or img_width != grt_width:
                raise Exception("source img and label img must has the same size")
        else:
            if mode == ModelPhase.TRAIN or mode == ModelPhase.EVAL:
                raise Exception("Empty image, src_dir: {}, img: {} & lab: {}".format(src_dir, img_path, grt_path))

        if len(img.shape) < 3:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        return img, grt_, img_name, grt_name
