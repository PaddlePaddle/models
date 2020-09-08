# -*- coding:utf-8 -*-
# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved
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
"""
data reader
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import os.path as osp
import signal
import numpy as np
import random
import logging
import math
import copy
import glob
import time
import re
from PIL import Image
import lib.util as util
from lib.rpn_util import *
import data.augmentations as augmentations

from easydict import EasyDict as edict
import cv2
import pdb

__all__ = ["M3drpnReader"]

logger = logging.getLogger(__name__)


class M3drpnReader(object):
    """m3drpn reader"""

    def __init__(self, conf, data_dir):
        self.data_dir = data_dir
        self.conf = conf
        self.video_det = False if not ('video_det' in conf) else conf.video_det
        self.video_count = 1 if not (
            'video_count' in conf) else conf.video_count
        self.use_3d_for_2d = ('use_3d_for_2d' in conf) and conf.use_3d_for_2d
        self.load_data()
        self.transform = augmentations.Augmentation(conf)

    def _read_data_file(self, fname):
        assert osp.isfile(fname), \
            "{} is not a file".format(fname)
        with open(fname) as f:
            return [line.strip() for line in f]

    def load_data(self):
        """load data"""
        logger.info("Loading KITTI dataset from {} ...".format(self.data_dir))
        # read all_files.txt

        for dbind, db in enumerate(self.conf.datasets_train):

            logging.info('Loading imgs_label {}'.format(db['name']))

            # single imdb
            imdb_single_db = []
            # kitti formatting
            if db['anno_fmt'].lower() == 'kitti_det':
                train_folder = os.path.join(self.data_dir, db['name'],
                                            'training')
                ann_folder = os.path.join(
                    train_folder, 'label_2',
                    '')  # dataset/kitti_split1/training/image_2/
                cal_folder = os.path.join(train_folder, 'calib', '')
                im_folder = os.path.join(train_folder, 'image_2', '')
                # get sorted filepaths
                annlist = sorted(glob(ann_folder + '*.txt'))  # 3712

                imdb_start = time()

                self.affine_size = None if not (
                    'affine_size' in self.conf) else self.conf.affine_size

                for annind, annpath in enumerate(annlist):
                    # get file parts
                    base = os.path.basename(annpath)
                    id, ext = os.path.splitext(base)

                    calpath = os.path.join(cal_folder, id + '.txt')
                    impath = os.path.join(im_folder, id + db['im_ext'])
                    impath_pre = os.path.join(train_folder, 'prev_2',
                                              id + '_01' + db['im_ext'])
                    impath_pre2 = os.path.join(train_folder, 'prev_2',
                                               id + '_02' + db['im_ext'])
                    impath_pre3 = os.path.join(train_folder, 'prev_2',
                                               id + '_03' + db['im_ext'])

                    # read gts
                    p2 = read_kitti_cal(calpath)
                    p2_inv = np.linalg.inv(p2)

                    gts = read_kitti_label(annpath, p2, self.use_3d_for_2d)

                    obj = edict()
                    # store gts
                    obj.id = id
                    obj.gts = gts
                    obj.p2 = p2
                    obj.p2_inv = p2_inv

                    # im properties
                    im = Image.open(impath)
                    obj.path = impath
                    obj.path_pre = impath_pre
                    obj.path_pre2 = impath_pre2
                    obj.path_pre3 = impath_pre3
                    obj.imW, obj.imH = im.size

                    # database properties
                    obj.dbname = db.name
                    obj.scale = db.scale
                    obj.dbind = dbind
                    obj.affine_gt = None  # did not compute transformer

                    # store
                    imdb_single_db.append(obj)

                    if (annind % 1000) == 0 and annind > 0:
                        time_str, dt = util.compute_eta(imdb_start, annind,
                                                        len(annlist))
                        logging.info('{}/{}, dt: {:0.4f}, eta: {}'.format(
                            annind, len(annlist), dt, time_str))
        self.data = {}
        self.data['train'] = imdb_single_db
        self.data['test'] = {}
        self.len = len(imdb_single_db)
        self.sampled_weights = balance_samples(self.conf, imdb_single_db)

    def _augmented_single(self, index):
        """
        Grabs the item at the given index. Specifically,
          - read the image from disk
          - read the imobj from RAM
          - applies data augmentation to (im, imobj)
          - converts image to RGB and [B C W H]
        """

        if not self.video_det:
            # read image
            im = cv2.imread(self.data['train'][index].path)
        else:
            # read images
            im = cv2.imread(self.data['train'][index].path)
            video_count = 1 if self.video_count is None else self.video_count

            if video_count >= 2:
                im_pre = cv2.imread(self.data['train'][index].path_pre)

                if not im_pre.shape == im.shape:
                    im_pre = cv2.resize(im_pre, (im.shape[1], im.shape[0]))

                im = np.concatenate((im, im_pre), axis=2)

            if video_count >= 3:

                im_pre2 = cv2.imread(self.data['train'][index].path_pre2)

                if im_pre2 is None:
                    im_pre2 = im_pre

                if not im_pre2.shape == im.shape:
                    im_pre2 = cv2.resize(im_pre2, (im.shape[1], im.shape[0]))

                im = np.concatenate((im, im_pre2), axis=2)

            if video_count >= 4:

                im_pre3 = cv2.imread(self.data['train'][index].path_pre3)

                if im_pre3 is None:
                    im_pre3 = im_pre2

                if not im_pre3.shape == im.shape:
                    im_pre3 = cv2.resize(im_pre3, (im.shape[1], im.shape[0]))

                im = np.concatenate((im, im_pre3), axis=2)

        # transform / data augmentation
        im, imobj = self.transform(im, copy.deepcopy(self.data['train'][index]))

        for i in range(int(im.shape[2] / 3)):
            # convert to RGB then permute to be [B C H W]
            im[:, :, (i * 3):(i * 3) + 3] = im[:, :, (i * 3 + 2, i * 3 + 1, i *
                                                      3)]
        im = np.transpose(im, [2, 0, 1])
        return im, imobj

    def get_reader(self, batch_size, mode='train', shuffle=True):
        """
        get reader
        """
        assert mode in ['train', 'test'], \
            "mode can only be 'train' or 'test'"
        imgs = self.data[mode]

        idxs = np.arange(len(imgs))

        idxs = np.random.choice(
            self.len, self.len, replace=True, p=self.sampled_weights)

        if mode == 'train' and shuffle:
            np.random.shuffle(idxs)

        def reader():
            """reader"""
            batch_out = []

            for ind in idxs:
                augmented_img, im_obj = self._augmented_single(ind)
                batch_out.append([augmented_img, im_obj])
                if len(batch_out) == batch_size:
                    yield batch_out
                    batch_out = []

        return reader


# derived from M3D-RPN
def read_kitti_cal(calfile):
    """
    Reads the kitti calibration projection matrix (p2) file from disc.
    Args:
        calfile (str): path to single calibration file
    """

    text_file = open(calfile, 'r')

    p2pat = re.compile((
        '(P2:)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)'
        + '\s+(fpat)\s+(fpat)\s+(fpat)\s*\n').replace(
            'fpat', '[-+]?[\d]+\.?[\d]*[Ee](?:[-+]?[\d]+)?'))

    for line in text_file:

        parsed = p2pat.fullmatch(line)

        # bbGt annotation in text format of:
        # cls x y w h occ x y w h ign ang
        if parsed is not None:
            p2 = np.zeros([4, 4], dtype=float)
            p2[0, 0] = parsed.group(2)
            p2[0, 1] = parsed.group(3)
            p2[0, 2] = parsed.group(4)
            p2[0, 3] = parsed.group(5)
            p2[1, 0] = parsed.group(6)
            p2[1, 1] = parsed.group(7)
            p2[1, 2] = parsed.group(8)
            p2[1, 3] = parsed.group(9)
            p2[2, 0] = parsed.group(10)
            p2[2, 1] = parsed.group(11)
            p2[2, 2] = parsed.group(12)
            p2[2, 3] = parsed.group(13)

            p2[3, 3] = 1

    text_file.close()

    return p2


def balance_samples(conf, imdb):
    """
    Balances the samples in an image dataset according to the given configuration.
    Basically we check which images have relevant foreground samples and which are empty,
    then we compute the sampling weights according to a desired fg_image_ratio.

    This is primarily useful in datasets which have a lot of empty (background) images, which may
    cause instability during training if not properly balanced against.
    """

    sample_weights = np.ones(len(imdb))

    if conf.fg_image_ratio >= 0:

        empty_inds = []
        valid_inds = []

        for imind, imobj in enumerate(imdb):

            valid = 0

            scale = conf.test_scale / imobj.imH
            igns, rmvs = determine_ignores(imobj.gts, conf.lbls, conf.ilbls,
                                           conf.min_gt_vis, conf.min_gt_h,
                                           conf.max_gt_h, scale)

            for gtind, gt in enumerate(imobj.gts):

                if (not igns[gtind]) and (not rmvs[gtind]):
                    valid += 1

            sample_weights[imind] = valid

            if valid > 0:
                valid_inds.append(imind)
            else:
                empty_inds.append(imind)

        if not (conf.fg_image_ratio == 2):
            fg_weight = 1
            bg_weight = 0
            sample_weights[valid_inds] = fg_weight
            sample_weights[empty_inds] = bg_weight

            logging.info('weighted respectively as {:.2f} and {:.2f}'.format(
                fg_weight, bg_weight))

        logging.info('Found {} foreground and {} empty images'.format(
            np.sum(sample_weights > 0), np.sum(sample_weights <= 0)))

    # force sampling weights to sum to 1
    sample_weights /= np.sum(sample_weights)
    return sample_weights


def read_kitti_poses(posefile):
    """
    read_kitti_poses
    """
    text_file = open(posefile, 'r')

    ppat1 = re.compile((
        '(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)'
        + '\s+(fpat)\s+(fpat)\s+(fpat)\s*\n').replace(
            'fpat', '[-+]?[\d]+\.?[\d]*[Ee](?:[-+]?[\d]+)?'))

    ppat2 = re.compile((
        '(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)'
        + '\s+(fpat)\s+(fpat)\s+(fpat)\s*\n').replace('fpat',
                                                      '[-+]?[\d]+\.?[\d]*'))

    ps = []

    for line in text_file:

        parsed1 = ppat1.fullmatch(line)
        parsed2 = ppat2.fullmatch(line)

        if parsed1 is not None:
            p = np.zeros([4, 4], dtype=float)
            p[0, 0] = parsed1.group(1)
            p[0, 1] = parsed1.group(2)
            p[0, 2] = parsed1.group(3)
            p[0, 3] = parsed1.group(4)
            p[1, 0] = parsed1.group(5)
            p[1, 1] = parsed1.group(6)
            p[1, 2] = parsed1.group(7)
            p[1, 3] = parsed1.group(8)
            p[2, 0] = parsed1.group(9)
            p[2, 1] = parsed1.group(10)
            p[2, 2] = parsed1.group(11)
            p[2, 3] = parsed1.group(12)
            p[3, 3] = 1

            ps.append(p)

        elif parsed2 is not None:

            p = np.zeros([4, 4], dtype=float)
            p[0, 0] = parsed2.group(1)
            p[0, 1] = parsed2.group(2)
            p[0, 2] = parsed2.group(3)
            p[0, 3] = parsed2.group(4)
            p[1, 0] = parsed2.group(5)
            p[1, 1] = parsed2.group(6)
            p[1, 2] = parsed2.group(7)
            p[1, 3] = parsed2.group(8)
            p[2, 0] = parsed2.group(9)
            p[2, 1] = parsed2.group(10)
            p[2, 2] = parsed2.group(11)
            p[2, 3] = parsed2.group(12)

            p[3, 3] = 1

            ps.append(p)

    text_file.close()

    return ps


def read_kitti_label(file, p2, use_3d_for_2d=False):
    """
    Reads the kitti label file from disc.
    Args:
        file (str): path to single label file for an image
        p2 (ndarray): projection matrix for the given image
    """

    gts = []

    text_file = open(file, 'r')
    '''
     Values    Name      Description
    ----------------------------------------------------------------------------
       1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                         'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                         'Misc' or 'DontCare'
       1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
                         truncated refers to the object leaving image boundaries
       1    occluded     Integer (0,1,2,3) indicating occlusion state:
                         0 = fully visible, 1 = partly occluded
                         2 = largely occluded, 3 = unknown
       1    alpha        Observation angle of object, ranging [-pi..pi]
       4    bbox         2D bounding box of object in the image (0-based index):
                         contains left, top, right, bottom pixel coordinates
       3    dimensions   3D object dimensions: height, width, length (in meters)
       3    location     3D object location x,y,z in camera coordinates (in meters)
       1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
       1    score        Only for results: Float, indicating confidence in
                         detection, needed for p/r curves, higher is better.
    '''

    pattern = re.compile((
        '([a-zA-Z\-\?\_]+)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+'
        +
        '(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s*((fpat)?)\n'
    ).replace('fpat', '[-+]?\d*\.\d+|[-+]?\d+'))

    for line in text_file:

        parsed = pattern.fullmatch(line)

        # bbGt annotation in text format of:
        # cls x y w h occ x y w h ign ang
        if parsed is not None:

            obj = edict()

            ign = False

            cls = parsed.group(1)
            trunc = float(parsed.group(2))
            occ = float(parsed.group(3))
            alpha = float(parsed.group(4))

            x = float(parsed.group(5))
            y = float(parsed.group(6))
            x2 = float(parsed.group(7))
            y2 = float(parsed.group(8))

            width = x2 - x + 1
            height = y2 - y + 1

            h3d = float(parsed.group(9))
            w3d = float(parsed.group(10))
            l3d = float(parsed.group(11))

            cx3d = float(parsed.group(12))  # center of car in 3d
            cy3d = float(parsed.group(13))  # bottom of car in 3d
            cz3d = float(parsed.group(14))  # center of car in 3d
            rotY = float(parsed.group(15))

            # actually center the box
            cy3d -= (h3d / 2)

            elevation = (1.65 - cy3d)

            if use_3d_for_2d and h3d > 0 and w3d > 0 and l3d > 0:

                # re-compute the 2D box using 3D (finally, avoids clipped boxes)
                verts3d, corners_3d = project_3d(
                    p2, cx3d, cy3d, cz3d, w3d, h3d, l3d, rotY, return_3d=True)

                # any boxes behind camera plane?
                if np.any(corners_3d[2, :] <= 0):
                    ign = True

                else:
                    x = min(verts3d[:, 0])
                    y = min(verts3d[:, 1])
                    x2 = max(verts3d[:, 0])
                    y2 = max(verts3d[:, 1])

                    width = x2 - x + 1
                    height = y2 - y + 1

            # project cx, cy, cz
            coord3d = p2.dot(np.array([cx3d, cy3d, cz3d, 1]))

            # store the projected instead
            cx3d_2d = coord3d[0]
            cy3d_2d = coord3d[1]
            cz3d_2d = coord3d[2]

            cx = cx3d_2d / cz3d_2d
            cy = cy3d_2d / cz3d_2d

            # encode occlusion with range estimation
            # 0 = fully visible, 1 = partly occluded
            # 2 = largely occluded, 3 = unknown
            if occ == 0:
                vis = 1
            elif occ == 1:
                vis = 0.66
            elif occ == 2:
                vis = 0.33
            else:
                vis = 0.0

            while rotY > math.pi:
                rotY -= math.pi * 2
            while rotY < (-math.pi):
                rotY += math.pi * 2

            # recompute alpha
            alpha = util.convertRot2Alpha(rotY, cz3d, cx3d)

            obj.elevation = elevation
            obj.cls = cls
            obj.occ = occ > 0
            obj.ign = ign
            obj.visibility = vis
            obj.trunc = trunc
            obj.alpha = alpha
            obj.rotY = rotY

            # is there an extra field? (assume to be track)
            if len(parsed.groups()) >= 16 and parsed.group(16).isdigit():
                obj.track = int(parsed.group(16))

            obj.bbox_full = np.array([x, y, width, height])
            obj.bbox_3d = [
                cx, cy, cz3d_2d, w3d, h3d, l3d, alpha, cx3d, cy3d, cz3d, rotY
            ]
            obj.center_3d = [cx3d, cy3d, cz3d]

            gts.append(obj)

    text_file.close()

    return gts


def _term_reader(signum, frame):
    """_term_reader"""
    logger.info('pid {} terminated, terminate reader process '
                'group {}...'.format(os.getpid(), os.getpgrp()))
    os.killpg(os.getpgid(os.getpid()), signal.SIGKILL)


signal.signal(signal.SIGINT, _term_reader)
signal.signal(signal.SIGTERM, _term_reader)
