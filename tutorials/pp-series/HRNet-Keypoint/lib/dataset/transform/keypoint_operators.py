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

try:
    from collections.abc import Sequence
except Exception:
    from collections import Sequence

import cv2
import numpy as np
import math
import copy

from lib.utils.keypoint_utils import warp_affine_joints, get_affine_transform, affine_transform, get_warp_matrix
from lib.utils.workspace import serializable
from lib.utils.logger import setup_logger
logger = setup_logger(__name__)

registered_ops = []

__all__ = [
    'RandomFlipHalfBodyTransform',
    'TopDownAffine',
    'ToHeatmapsTopDown',
    'TopDownEvalAffine',
]


def register_keypointop(cls):
    return serializable(cls)


@register_keypointop
class RandomFlipHalfBodyTransform(object):
    """apply data augment to image and coords
    to achieve the flip, scale, rotate and half body transform effect for training image

    Args:
        trainsize (list):[w, h], Image target size
        upper_body_ids (list): The upper body joint ids
        flip_pairs (list): The left-right joints exchange order list
        pixel_std (int): The pixel std of the scale
        scale (float): The scale factor to transform the image
        rot (int): The rotate factor to transform the image
        num_joints_half_body (int): The joints threshold of the half body transform
        prob_half_body (float): The threshold of the half body transform
        flip (bool): Whether to flip the image

    Returns:
        records(dict): contain the image and coords after tranformed

    """

    def __init__(self,
                 trainsize,
                 upper_body_ids,
                 flip_pairs,
                 pixel_std,
                 scale=0.35,
                 rot=40,
                 num_joints_half_body=8,
                 prob_half_body=0.3,
                 flip=True,
                 rot_prob=0.6):
        super(RandomFlipHalfBodyTransform, self).__init__()
        self.trainsize = trainsize
        self.upper_body_ids = upper_body_ids
        self.flip_pairs = flip_pairs
        self.pixel_std = pixel_std
        self.scale = scale
        self.rot = rot
        self.num_joints_half_body = num_joints_half_body
        self.prob_half_body = prob_half_body
        self.flip = flip
        self.aspect_ratio = trainsize[0] * 1.0 / trainsize[1]
        self.rot_prob = rot_prob

    def halfbody_transform(self, joints, joints_vis):
        upper_joints = []
        lower_joints = []
        for joint_id in range(joints.shape[0]):
            if joints_vis[joint_id][0] > 0:
                if joint_id in self.upper_body_ids:
                    upper_joints.append(joints[joint_id])
                else:
                    lower_joints.append(joints[joint_id])
        if np.random.randn() < 0.5 and len(upper_joints) > 2:
            selected_joints = upper_joints
        else:
            selected_joints = lower_joints if len(
                lower_joints) > 2 else upper_joints
        if len(selected_joints) < 2:
            return None, None
        selected_joints = np.array(selected_joints, dtype=np.float32)
        center = selected_joints.mean(axis=0)[:2]
        left_top = np.amin(selected_joints, axis=0)
        right_bottom = np.amax(selected_joints, axis=0)
        w = right_bottom[0] - left_top[0]
        h = right_bottom[1] - left_top[1]
        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array(
            [w * 1.0 / self.pixel_std, h * 1.0 / self.pixel_std],
            dtype=np.float32)
        scale = scale * 1.5

        return center, scale

    def flip_joints(self, joints, joints_vis, width, matched_parts):
        joints[:, 0] = width - joints[:, 0] - 1
        for pair in matched_parts:
            joints[pair[0], :], joints[pair[1], :] = \
                joints[pair[1], :], joints[pair[0], :].copy()
            joints_vis[pair[0], :], joints_vis[pair[1], :] = \
                joints_vis[pair[1], :], joints_vis[pair[0], :].copy()

        return joints * joints_vis, joints_vis

    def __call__(self, records):
        image = records['image']
        joints = records['joints']
        joints_vis = records['joints_vis']
        c = records['center']
        s = records['scale']
        r = 0
        if (np.sum(joints_vis[:, 0]) > self.num_joints_half_body and
                np.random.rand() < self.prob_half_body):
            c_half_body, s_half_body = self.halfbody_transform(joints,
                                                               joints_vis)
            if c_half_body is not None and s_half_body is not None:
                c, s = c_half_body, s_half_body
        sf = self.scale
        rf = self.rot
        s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
        r = np.clip(np.random.randn() * rf, -rf * 2,
                    rf * 2) if np.random.random() <= self.rot_prob else 0

        if self.flip and np.random.random() <= 0.5:
            image = image[:, ::-1, :]
            joints, joints_vis = self.flip_joints(
                joints, joints_vis, image.shape[1], self.flip_pairs)
            c[0] = image.shape[1] - c[0] - 1
        records['image'] = image
        records['joints'] = joints
        records['joints_vis'] = joints_vis
        records['center'] = c
        records['scale'] = s
        records['rotate'] = r

        return records


@register_keypointop
class TopDownAffine(object):
    """apply affine transform to image and coords

    Args:
        trainsize (list): [w, h], the standard size used to train
        use_udp (bool): whether to use Unbiased Data Processing.
        records(dict): the dict contained the image and coords

    Returns:
        records (dict): contain the image and coords after tranformed

    """

    def __init__(self, trainsize, use_udp=False):
        self.trainsize = trainsize
        self.use_udp = use_udp

    def __call__(self, records):
        image = records['image']
        joints = records['joints']
        joints_vis = records['joints_vis']
        rot = records['rotate'] if "rotate" in records else 0
        if self.use_udp:
            trans = get_warp_matrix(
                rot, records['center'] * 2.0,
                [self.trainsize[0] - 1.0, self.trainsize[1] - 1.0],
                records['scale'] * 200.0)
            image = cv2.warpAffine(
                image,
                trans, (int(self.trainsize[0]), int(self.trainsize[1])),
                flags=cv2.INTER_LINEAR)
            joints[:, 0:2] = warp_affine_joints(joints[:, 0:2].copy(), trans)
        else:
            trans = get_affine_transform(records['center'], records['scale'] *
                                         200, rot, self.trainsize)
            image = cv2.warpAffine(
                image,
                trans, (int(self.trainsize[0]), int(self.trainsize[1])),
                flags=cv2.INTER_LINEAR)
            for i in range(joints.shape[0]):
                if joints_vis[i, 0] > 0.0:
                    joints[i, 0:2] = affine_transform(joints[i, 0:2], trans)

        records['image'] = image
        records['joints'] = joints

        return records


@register_keypointop
class TopDownEvalAffine(object):
    """apply affine transform to image and coords

    Args:
        trainsize (list): [w, h], the standard size used to train
        use_udp (bool): whether to use Unbiased Data Processing.
        records(dict): the dict contained the image and coords

    Returns:
        records (dict): contain the image and coords after tranformed

    """

    def __init__(self, trainsize, use_udp=False):
        self.trainsize = trainsize
        self.use_udp = use_udp

    def __call__(self, records):
        image = records['image']
        rot = 0
        imshape = records['im_shape'][::-1]
        center = imshape / 2.
        scale = imshape

        if self.use_udp:
            trans = get_warp_matrix(
                rot, center * 2.0,
                [self.trainsize[0] - 1.0, self.trainsize[1] - 1.0], scale)
            image = cv2.warpAffine(
                image,
                trans, (int(self.trainsize[0]), int(self.trainsize[1])),
                flags=cv2.INTER_LINEAR)
        else:
            trans = get_affine_transform(center, scale, rot, self.trainsize)
            image = cv2.warpAffine(
                image,
                trans, (int(self.trainsize[0]), int(self.trainsize[1])),
                flags=cv2.INTER_LINEAR)
        records['image'] = image

        return records


@register_keypointop
class ToHeatmapsTopDown(object):
    """to generate the gaussin heatmaps of keypoint for heatmap loss

    Args:
        hmsize (list): [w, h] output heatmap's size
        sigma (float): the std of gaussin kernel genereted
        records(dict): the dict contained the image and coords

    Returns:
        records (dict): contain the heatmaps used to heatmaploss

    """

    def __init__(self, hmsize, sigma):
        super(ToHeatmapsTopDown, self).__init__()
        self.hmsize = np.array(hmsize)
        self.sigma = sigma

    def __call__(self, records):
        joints = records['joints']
        joints_vis = records['joints_vis']
        num_joints = joints.shape[0]
        image_size = np.array(
            [records['image'].shape[1], records['image'].shape[0]])
        target_weight = np.ones((num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = joints_vis[:, 0]
        target = np.zeros(
            (num_joints, self.hmsize[1], self.hmsize[0]), dtype=np.float32)
        tmp_size = self.sigma * 3
        feat_stride = image_size / self.hmsize
        for joint_id in range(num_joints):
            mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
            mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
            # Check that any part of the gaussian is in-bounds
            ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
            br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
            if ul[0] >= self.hmsize[0] or ul[1] >= self.hmsize[1] or br[
                    0] < 0 or br[1] < 0:
                # If not, just return the image as is
                target_weight[joint_id] = 0
                continue
            # # Generate gaussian
            size = 2 * tmp_size + 1
            x = np.arange(0, size, 1, np.float32)
            y = x[:, np.newaxis]
            x0 = y0 = size // 2
            # The gaussian is not normalized, we want the center value to equal 1
            g = np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * self.sigma**2))

            # Usable gaussian range
            g_x = max(0, -ul[0]), min(br[0], self.hmsize[0]) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], self.hmsize[1]) - ul[1]
            # Image range
            img_x = max(0, ul[0]), min(br[0], self.hmsize[0])
            img_y = max(0, ul[1]), min(br[1], self.hmsize[1])

            v = target_weight[joint_id]
            if v > 0.5:
                target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[
                    0]:g_y[1], g_x[0]:g_x[1]]
        records['target'] = target
        records['target_weight'] = target_weight
        del records['joints'], records['joints_vis']

        return records
