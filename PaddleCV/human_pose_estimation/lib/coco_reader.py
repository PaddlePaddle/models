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
"""Data reader for COCO dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import functools
import numpy as np
import cv2
import random

from utils.transforms import fliplr_joints
from utils.transforms import get_affine_transform
from utils.transforms import affine_transform
from lib.base_reader import visualize, generate_target
from pycocotools.coco import COCO

# NOTE
# -- COCO Datatset --
# "keypoints": 
# {
#   0: "nose",
#   1: "left_eye",
#   2: "right_eye",
#   3: "left_ear",
#   4: "right_ear",
#   5: "left_shoulder",
#   6: "right_shoulder",
#   7: "left_elbow",
#   8: "right_elbow",
#   9: "left_wrist",
#   10: "right_wrist",
#   11: "left_hip",
#   12: "right_hip",
#   13: "left_knee",
#   14: "right_knee",
#   15: "left_ankle",
#   16: "right_ankle"
# },
#
# "skeleton": 
# [
#   [16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13], [6,7],[6,8],
#   [7,9],[8,10],[9,11],[2,3],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7]
# ]


class Config:
    """Configurations for COCO dataset.
    """
    DEBUG = False
    TMPDIR = 'tmp_fold_for_debug'

    # For reader
    BUF_SIZE = 102400
    THREAD = 1 if DEBUG else 8  # have to be larger than 0

    # Fixed infos of dataset
    DATAROOT = 'data/coco'
    IMAGEDIR = 'images'
    NUM_JOINTS = 17
    FLIP_PAIRS = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14],
                  [15, 16]]
    PARENT_IDS = None

    # CFGS
    SCALE_FACTOR = 0.3
    ROT_FACTOR = 40
    FLIP = True
    TARGET_TYPE = 'gaussian'
    SIGMA = 3
    IMAGE_SIZE = [288, 384]
    HEATMAP_SIZE = [72, 96]
    ASPECT_RATIO = IMAGE_SIZE[0] * 1.0 / IMAGE_SIZE[1]
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    PIXEL_STD = 200


cfg = Config()


def _box2cs(box):
    x, y, w, h = box[:4]
    return _xywh2cs(x, y, w, h)


def _xywh2cs(x, y, w, h):
    center = np.zeros((2), dtype=np.float32)
    center[0] = x + w * 0.5
    center[1] = y + h * 0.5

    if w > cfg.ASPECT_RATIO * h:
        h = w * 1.0 / cfg.ASPECT_RATIO
    elif w < cfg.ASPECT_RATIO * h:
        w = h * cfg.ASPECT_RATIO
    scale = np.array(
        [w * 1.0 / cfg.PIXEL_STD, h * 1.0 / cfg.PIXEL_STD], dtype=np.float32)
    if center[0] != -1:
        scale = scale * 1.25

    return center, scale


def _select_data(db):
    db_selected = []
    for rec in db:
        num_vis = 0
        joints_x = 0.0
        joints_y = 0.0
        for joint, joint_vis in zip(rec['joints_3d'], rec['joints_3d_vis']):
            if joint_vis[0] <= 0:
                continue
            num_vis += 1

            joints_x += joint[0]
            joints_y += joint[1]
        if num_vis == 0:
            continue

        joints_x, joints_y = joints_x / num_vis, joints_y / num_vis

        area = rec['scale'][0] * rec['scale'][1] * (cfg.PIXEL_STD**2)
        joints_center = np.array([joints_x, joints_y])
        bbox_center = np.array(rec['center'])
        diff_norm2 = np.linalg.norm((joints_center - bbox_center), 2)
        ks = np.exp(-1.0 * (diff_norm2**2) / ((0.2)**2 * 2.0 * area))

        metric = (0.2 / 16) * num_vis + 0.45 - 0.2 / 16
        if ks > metric:
            db_selected.append(rec)

    print('=> num db: {}'.format(len(db)))
    print('=> num selected db: {}'.format(len(db_selected)))
    return db_selected


def _load_coco_keypoint_annotation(image_set_index, coco,
                                   _coco_ind_to_class_ind, image_set):
    """Ground truth bbox and keypoints.
    """
    print('generating coco gt_db...')
    gt_db = []
    for index in image_set_index:
        im_ann = coco.loadImgs(index)[0]
        width = im_ann['width']
        height = im_ann['height']

        annIds = coco.getAnnIds(imgIds=index, iscrowd=False)
        objs = coco.loadAnns(annIds)

        # Sanitize bboxes
        valid_objs = []
        for obj in objs:
            x, y, w, h = obj['bbox']
            x1 = np.max((0, x))
            y1 = np.max((0, y))
            x2 = np.min((width - 1, x1 + np.max((0, w - 1))))
            y2 = np.min((height - 1, y1 + np.max((0, h - 1))))
            if obj['area'] > 0 and x2 >= x1 and y2 >= y1:
                obj['clean_bbox'] = [x1, y1, x2 - x1, y2 - y1]
                valid_objs.append(obj)
        objs = valid_objs

        rec = []
        for obj in objs:
            cls = _coco_ind_to_class_ind[obj['category_id']]
            if cls != 1:
                continue

            # Ignore objs without keypoints annotation
            if max(obj['keypoints']) == 0:
                continue

            joints_3d = np.zeros((cfg.NUM_JOINTS, 3), dtype=np.float)
            joints_3d_vis = np.zeros((cfg.NUM_JOINTS, 3), dtype=np.float)
            for ipt in range(cfg.NUM_JOINTS):
                joints_3d[ipt, 0] = obj['keypoints'][ipt * 3 + 0]
                joints_3d[ipt, 1] = obj['keypoints'][ipt * 3 + 1]
                joints_3d[ipt, 2] = 0
                t_vis = obj['keypoints'][ipt * 3 + 2]
                if t_vis > 1:
                    t_vis = 1
                joints_3d_vis[ipt, 0] = t_vis
                joints_3d_vis[ipt, 1] = t_vis
                joints_3d_vis[ipt, 2] = 0

            center, scale = _box2cs(obj['clean_bbox'][:4])
            rec.append({
                'image': os.path.join(cfg.DATAROOT, cfg.IMAGEDIR,
                                      image_set + '2017', '%012d.jpg' % index),
                'center': center,
                'scale': scale,
                'joints_3d': joints_3d,
                'joints_3d_vis': joints_3d_vis,
                'filename': '%012d.jpg' % index,
                'imgnum': 0,
            })

        gt_db.extend(rec)
    return gt_db


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

    # used for ce
    if 'ce_mode' in os.environ:
        random.seed(0)
        np.random.seed(0)

    data_numpy = cv2.imread(image_file, cv2.IMREAD_COLOR |
                            cv2.IMREAD_IGNORE_ORIENTATION)

    if is_train:
        sf = cfg.SCALE_FACTOR
        rf = cfg.ROT_FACTOR
        s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
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
        trans, (int(cfg.IMAGE_SIZE[0]), int(cfg.IMAGE_SIZE[1])),
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
        return input, target, target_weight, c, s, score, image_file


# Create a reader
def _reader_creator(root,
                    image_set,
                    shuffle=False,
                    is_train=False,
                    use_gt_bbox=False):
    def reader():
        if image_set in ['train', 'val']:
            file_name = os.path.join(
                root, 'annotations',
                'person_keypoints_' + image_set + '2017.json')
        elif image_set in ['test', 'test-dev']:
            file_name = os.path.join(root, 'annotations',
                                     'image_info_' + image_set + '2017.json')
        else:
            raise ValueError("The dataset '{}' is not supported".format(
                image_set))

        # Load annotations
        coco = COCO(file_name)

        # Deal with class names
        cats = [cat['name'] for cat in coco.loadCats(coco.getCatIds())]
        classes = ['__background__'] + cats
        print('=> classes: {}'.format(classes))
        num_classes = len(classes)
        _class_to_ind = dict(zip(classes, range(num_classes)))
        _class_to_coco_ind = dict(zip(cats, coco.getCatIds()))
        _coco_ind_to_class_ind = dict([(_class_to_coco_ind[cls],
                                        _class_to_ind[cls])
                                       for cls in classes[1:]])

        # Load image file names
        image_set_index = coco.getImgIds()
        num_images = len(image_set_index)
        print('=> num_images: {}'.format(num_images))

        if is_train or use_gt_bbox:
            gt_db = _load_coco_keypoint_annotation(
                image_set_index, coco, _coco_ind_to_class_ind, image_set)
            gt_db = _select_data(gt_db)

        if shuffle:
            random.shuffle(gt_db)

        for db in gt_db:
            yield db

    mapper = functools.partial(data_augmentation, is_train=is_train)
    return reader, mapper


def train():
    reader, mapper = _reader_creator(
        cfg.DATAROOT, 'train', shuffle=True, is_train=True)

    # used for ce
    if 'ce_mode' in os.environ:
        reader, mapper = _reader_creator(
            cfg.DATAROOT, 'train', shuffle=False, is_train=True)

    def pop():
        for i, x in enumerate(reader()):
            yield mapper(x)

    return pop


def valid():
    reader, mapper = _reader_creator(
        cfg.DATAROOT, 'val', shuffle=False, is_train=False, use_gt_bbox=True)

    def pop():
        for i, x in enumerate(reader()):
            yield mapper(x)

    return pop


def test():
    reader, mapper = _reader_creator(
        cfg.DATAROOT, 'test', shuffle=False, is_train=False, use_gt_bbox=True)

    def pop():
        for i, x in enumerate(reader()):
            yield mapper(x)

    return pop
