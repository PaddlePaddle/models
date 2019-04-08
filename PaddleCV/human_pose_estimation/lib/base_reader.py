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

"""Libs for data reader."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import cv2
import numpy as np

def visualize(cfg, filename, data_numpy, input, joints, target):
    """
    :param cfg: global configurations for dataset
    :param filename: the name of image file
    :param data_numpy: original numpy image data
    :param input: input tensor [b, c, h, w]
    :param joints: [num_joints, 3]
    :param target: target tensor [b, c, h, w]
    """
    TMPDIR = cfg.TMPDIR
    NUM_JOINTS = cfg.NUM_JOINTS

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

def generate_target(cfg, joints, joints_vis):
    """
    :param joints:  [num_joints, 3]
    :param joints_vis: [num_joints, 3]
    :return: target, target_weight(1: visible, 0: invisible)
    """
    NUM_JOINTS = cfg.NUM_JOINTS
    TARGET_TYPE = cfg.TARGET_TYPE
    HEATMAP_SIZE = cfg.HEATMAP_SIZE
    IMAGE_SIZE = cfg.IMAGE_SIZE
    SIGMA = cfg.SIGMA

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

            # Generate gaussian
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
