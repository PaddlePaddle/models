# Copyright (c) 2018-present, Baidu, Inc.
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
##############################################################################

"""Data reader for MPII."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import functools
import json
import numpy as np
import cv2

from utils.transforms import fliplr_joints
from utils.transforms import get_affine_transform
from utils.transforms import affine_transform
from lib.base_reader import visualize, generate_target

class Config:
    """Configurations for MPII dataset.
    """
    DEBUG = False
    TMPDIR = 'tmp_fold_for_debug'

    # For reader
    BUF_SIZE = 102400
    THREAD = 1 if DEBUG else 8 # have to be larger than 0

    # Fixed infos of dataset
    DATAROOT = 'data/mpii'
    IMAGEDIR = 'images'
    NUM_JOINTS = 16
    FLIP_PAIRS = [[0, 5], [1, 4], [2, 3], [10, 15], [11, 14], [12, 13]]
    PARENT_IDS = [1, 2, 6, 6, 3, 4, 6, 6, 7, 8, 11, 12, 7, 7, 13, 14]

    # CFGS
    SCALE_FACTOR = 0.3
    ROT_FACTOR = 40
    FLIP = True
    TARGET_TYPE = 'gaussian'
    SIGMA = 3
    IMAGE_SIZE = [384, 384]
    HEATMAP_SIZE = [96, 96]
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

cfg = Config()

def data_augmentation(sample, is_train):
    image_file = sample['image']
    filename = sample['filename'] if 'filename' in sample else ''
    joints = sample['joints_3d']
    joints_vis = sample['joints_3d_vis']
    c = sample['center']
    s = sample['scale']
    score = sample['score'] if 'score' in sample else 1
    # imgnum = sample['imgnum'] if 'imgnum' in sample else ''
    r = 0

    data_numpy = cv2.imread(
        image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

    if is_train:
        sf = cfg.SCALE_FACTOR
        rf = cfg.ROT_FACTOR
        s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
        r = np.clip(np.random.randn()*rf, -rf*2, rf*2) \
                if random.random() <= 0.6 else 0

        if cfg.FLIP and random.random() <= 0.5:
            data_numpy = data_numpy[:, ::-1, :]
            joints, joints_vis = fliplr_joints(
                    joints, joints_vis, data_numpy.shape[1], cfg.FLIP_PAIRS)
            c[0] = data_numpy.shape[1] - c[0] - 1

    trans = get_affine_transform(c, s, r, cfg.IMAGE_SIZE)
    input = cv2.warpAffine(
            data_numpy,
            trans,
            (int(cfg.IMAGE_SIZE[0]), int(cfg.IMAGE_SIZE[1])),
            flags=cv2.INTER_LINEAR)

    for i in range(cfg.NUM_JOINTS):
        if joints_vis[i, 0] > 0.0:
            joints[i, 0:2] = affine_transform(joints[i, 0:2], trans)

    # Numpy target
    target, target_weight = generate_target(cfg, joints, joints_vis)

    if cfg.DEBUG:
        visualize(cfg, filename, data_numpy, input.copy(), joints, target)

    # Normalization
    input = input.astype('float32').transpose((2, 0, 1)) / 255
    input -= np.array(cfg.MEAN).reshape((3, 1, 1))
    input /= np.array(cfg.STD).reshape((3, 1, 1))

    if is_train:
        return input, target, target_weight
    else:
        return input, target, target_weight, c, s, score

def test_data_augmentation(sample):
    image_file = sample['image']
    filename = sample['filename'] if 'filename' in sample else ''

    file_id = int(filename.split('.')[0])

    input = cv2.imread(
            image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

    input = cv2.resize(input, (int(cfg.IMAGE_SIZE[0]), int(cfg.IMAGE_SIZE[1])))

    # Normalization
    input = input.astype('float32').transpose((2, 0, 1)) / 255
    input -= np.array(cfg.MEAN).reshape((3, 1, 1))
    input /= np.array(cfg.STD).reshape((3, 1, 1))

    return input, file_id

# Create a reader
def _reader_creator(root, image_set, shuffle=False, is_train=False):
    def reader():
        if image_set != 'test':
            file_name = os.path.join(root, 'annot', image_set+'.json')
            with open(file_name) as anno_file:
                anno = json.load(anno_file)
            print('=> load {} samples of {} dataset'.format(len(anno), image_set))

            if shuffle:
                random.shuffle(anno)

            for a in anno:
                image_name = a['image']

                c = np.array(a['center'], dtype=np.float)
                s = np.array([a['scale'], a['scale']], dtype=np.float)

                # Adjust center/scale slightly to avoid cropping limbs
                if c[0] != -1:
                    c[1] = c[1] + 15 * s[1]
                    s = s * 1.25

                # MPII uses matlab format, index is based 1,
                # we should first convert to 0-based index
                c = c - 1

                joints_3d = np.zeros((cfg.NUM_JOINTS, 3), dtype=np.float)
                joints_3d_vis = np.zeros((cfg.NUM_JOINTS, 3), dtype=np.float)

                joints = np.array(a['joints'])
                joints[:, 0:2] = joints[:, 0:2] - 1
                joints_vis = np.array(a['joints_vis'])
                assert len(joints) == cfg.NUM_JOINTS, \
                        'joint num diff: {} vs {}'.format(len(joints), cfg.NUM_JOINTS)

                joints_3d[:, 0:2] = joints[:, 0:2]
                joints_3d_vis[:, 0] = joints_vis[:]
                joints_3d_vis[:, 1] = joints_vis[:]

                yield dict(
                        image=os.path.join(cfg.DATAROOT, cfg.IMAGEDIR, image_name),
                        center=c,
                        scale=s,
                        joints_3d=joints_3d,
                        joints_3d_vis=joints_3d_vis,
                        filename=image_name,
                        test_mode=False,
                        imagenum=0)
        else:
            fold = os.path.join(cfg.DATAROOT, cfg.IMAGEDIR, 'test')
            for img_name in os.listdir(fold):
                yield dict(image=os.path.join(fold, img_name),
                           filename=img_name)

    if not image_set == 'test':
        mapper = functools.partial(data_augmentation, is_train=is_train)
    else:
        mapper = functools.partial(test_data_augmentation)
    return reader, mapper

def train():
    reader, mapper = _reader_creator(cfg.DATAROOT, 'train', shuffle=True, is_train=True)
    def pop():
         for i, x in enumerate(reader()):
             yield mapper(x)
    return pop

def valid():
    reader, mapper = _reader_creator(cfg.DATAROOT, 'valid', shuffle=False, is_train=False)
    def pop():
        for i, x in enumerate(reader()):
            yield mapper(x)
    return pop

def test():
    reader, mapper = _reader_creator(cfg.DATAROOT, 'test')
    def pop():
        for i, x in enumerate(reader()):
            yield mapper(x)
    return pop
