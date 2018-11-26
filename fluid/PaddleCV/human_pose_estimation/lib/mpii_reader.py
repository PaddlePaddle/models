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
import functools
import json
import numpy as np
import cv2
import random
import shutil

from utils.transforms import fliplr_joints
from utils.transforms import get_affine_transform
from utils.transforms import affine_transform

DEBUG = False
TMPDIR = 'tmp'

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

# Data augmentation
def _visualize(filename, data_numpy, input, joints, target):
    if os.path.exists(TMPDIR):
        shutil.rmtree(TMPDIR)
        os.mkdir(TMPDIR)
    else:
        os.mkdir(TMPDIR)

    f = open(os.path.join(TMPDIR, filename), 'w')
    f.close()

    cv2.imwrite(os.path.join(TMPDIR, 'flip.jpg'), data_numpy)
    cv2.imwrite(os.path.join(TMPDIR, 'input.jpg'), input)
    for i in range(NUM_JOINTS):
        cv2.imwrite(os.path.join(TMPDIR, 'target_{}.jpg'.format(i)), cv2.applyColorMap(
            np.uint8(np.expand_dims(target[i], 2)*255.), cv2.COLORMAP_JET))
        cv2.circle(input, (int(joints[i, 0]), int(joints[i, 1])), 5, [170, 255, 0], -1)
    cv2.imwrite(os.path.join(TMPDIR, 'input_kps.jpg'), input)

def _generate_target(joints, joints_vis):
    """
    :param joints:  [num_joints, 3]
    :param joints_vis: [num_joints, 3]
    :return: target, target_weight(1: visible, 0: invisible)
    """
    target_weight = np.ones((NUM_JOINTS, 1), dtype=np.float32)
    target_weight[:, 0] = joints_vis[:, 0]

    assert TARGET_TYPE == 'gaussian', \
        'Only support gaussian map now!'

    if TARGET_TYPE == 'gaussian':
        target = np.zeros((NUM_JOINTS,
                           HEATMAP_SIZE[1],
                           HEATMAP_SIZE[0]),
                           dtype=np.float32)

        tmp_size = SIGMA * 3

        for joint_id in range(NUM_JOINTS):
            feat_stride = np.array(IMAGE_SIZE) / np.array(HEATMAP_SIZE)
            mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
            mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)

            # Check that any part of the gaussian is in-bounds
            ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
            br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
            if ul[0] >= HEATMAP_SIZE[0] or ul[1] >= HEATMAP_SIZE[1] \
                    or br[0] < 0 or br[1] < 0:
                # If not, just return the image as is
                target_weight[joint_id] = 0
                continue

            # # Generate gaussian
            size = 2 * tmp_size + 1
            x = np.arange(0, size, 1, np.float32)
            y = x[:, np.newaxis]
            x0 = y0 = size // 2
            # The gaussian is not normalized, we want the center value to equal 1
            g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * SIGMA ** 2))

            # Usable gaussian range
            g_x = max(0, -ul[0]), min(br[0], HEATMAP_SIZE[0]) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], HEATMAP_SIZE[1]) - ul[1]
            # Image range
            img_x = max(0, ul[0]), min(br[0], HEATMAP_SIZE[0])
            img_y = max(0, ul[1]), min(br[1], HEATMAP_SIZE[1])

            v = target_weight[joint_id]
            if v > 0.5:
                target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                    g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

    return target, target_weight

def data_augmentation(sample, is_train):
    image_file = sample['image']
    filename = sample['filename'] if 'filename' in sample else ''
    # imgnum = sample['imgnum'] if 'imgnum' in sample else ''

    data_numpy = cv2.imread(
            image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

    joints = sample['joints_3d']
    joints_vis = sample['joints_3d_vis']
    c = sample['center']
    s = sample['scale']
    score = sample['score'] if 'score' in sample else 1
    r = 0

    if is_train:
        sf = SCALE_FACTOR
        rf = ROT_FACTOR
        s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
        r = np.clip(np.random.randn()*rf, -rf*2, rf*2) \
                if random.random() <= 0.6 else 0

        if FLIP and random.random() <= 0.6:
            data_numpy = data_numpy[:, ::-1, :]
            joints, joints_vis = fliplr_joints(
                    joints, joints_vis, data_numpy.shape[1], FLIP_PAIRS)
            c[0] = data_numpy.shape[1] - c[0] - 1

    trans = get_affine_transform(c, s, r, IMAGE_SIZE)
    input = cv2.warpAffine(
            data_numpy,
            trans,
            (int(IMAGE_SIZE[0]), int(IMAGE_SIZE[1])),
            flags=cv2.INTER_LINEAR)

    for i in range(NUM_JOINTS):
        if joints_vis[i, 0] > 0.0:
            joints[i, 0:2] = affine_transform(joints[i, 0:2], trans)

    # Numpy target
    target, target_weight = _generate_target(joints, joints_vis)

    if DEBUG:
        _visualize(filename, data_numpy, input.copy(), joints, target)

    # input_no_norm = input.copy()

    # Normalization
    input = input.astype('float32').transpose((2, 0, 1)) / 255
    input -= np.array(MEAN).reshape((3, 1, 1))
    input /= np.array(STD).reshape((3, 1, 1))

    if is_train:
        return input, target, target_weight
    else:
        return input, target, target_weight, c, s, score

def test_data_augmentation(sample):
    image_file = sample['image']
    filename = sample['filename'] if 'filename' in sample else ''

    file_id = int(filename.split('.')[0].split('_')[1])

    input = cv2.imread(
            image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

    input = cv2.resize(input, (int(IMAGE_SIZE[0]), int(IMAGE_SIZE[1])))

    # Normalization
    input = input.astype('float32').transpose((2, 0, 1)) / 255
    input -= np.array(MEAN).reshape((3, 1, 1))
    input /= np.array(STD).reshape((3, 1, 1))

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

                joints_3d = np.zeros((NUM_JOINTS, 3), dtype=np.float)
                joints_3d_vis = np.zeros((NUM_JOINTS, 3), dtype=np.float)

                joints = np.array(a['joints'])
                joints[:, 0:2] = joints[:, 0:2] - 1
                joints_vis = np.array(a['joints_vis'])
                assert len(joints) == NUM_JOINTS, \
                        'joint num diff: {} vs {}'.format(len(joints), NUM_JOINTS)

                joints_3d[:, 0:2] = joints[:, 0:2]
                joints_3d_vis[:, 0] = joints_vis[:]
                joints_3d_vis[:, 1] = joints_vis[:]

                yield dict(
                        image = os.path.join(DATAROOT, IMAGEDIR, image_name),
                        center = c,
                        scale = s,
                        joints_3d = joints_3d,
                        joints_3d_vis = joints_3d_vis,
                        filename = image_name,
                        test_mode = False,
                        imagenum = 0)


        else:
            fold = 'test'
            for img_name in os.listdir(fold):
                yield dict(image = os.path.join(fold, img_name),
                           filename = img_name)

    if not image_set == 'test':
        mapper = functools.partial(data_augmentation, is_train=is_train)
    else:
        mapper = functools.partial(test_data_augmentation)
    return reader, mapper

def train():
    reader, mapper = _reader_creator(DATAROOT, 'train', shuffle=True, is_train=True)
    def pop():
         for i, x in enumerate(reader()):
             yield mapper(x)
    return pop

def valid():
    reader, mapper = _reader_creator(DATAROOT, 'valid', shuffle=False, is_train=False)
    def pop():
        for i, x in enumerate(reader()):
            yield mapper(x)
    return pop

def test():
    reader, mapper = _reader_creator(DATAROOT, 'test')
    def pop():
        for i, x in enumerate(reader()):
            yield mapper(x)
    return pop

