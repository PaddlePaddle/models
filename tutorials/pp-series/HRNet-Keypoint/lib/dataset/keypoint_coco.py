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
from __future__ import division
from __future__ import print_function

import os
import cv2
import numpy as np
import json
import copy
import pycocotools
from pycocotools.coco import COCO
from .dataset import DetDataset
from lib.utils.workspace import register, serializable

__all__ = ['KeypointTopDownBaseDataset', 'KeypointTopDownCocoDataset']


@serializable
class KeypointTopDownBaseDataset(DetDataset):
    """Base class for top_down datasets.

    All datasets should subclass it.
    All subclasses should overwrite:
        Methods:`_get_db`

    Args:
        dataset_dir (str): Root path to the dataset.
        image_dir (str): Path to a directory where images are held.
        anno_path (str): Relative path to the annotation file.
        num_joints (int): keypoint numbers
        transform (composed(operators)): A sequence of data transforms.
    """

    def __init__(self,
                 dataset_dir,
                 image_dir,
                 anno_path,
                 num_joints,
                 transform=[]):
        super().__init__(dataset_dir, image_dir, anno_path)
        self.image_info = {}
        self.ann_info = {}

        self.img_prefix = os.path.join(dataset_dir, image_dir)
        self.transform = transform

        self.ann_info['num_joints'] = num_joints
        self.db = []

    def __len__(self):
        """Get dataset length."""
        return len(self.db)

    def _get_db(self):
        """Get a sample"""
        raise NotImplementedError

    def __getitem__(self, idx):
        """Prepare sample for training given the index."""
        records = copy.deepcopy(self.db[idx])
        records['image'] = cv2.imread(records['image_file'], cv2.IMREAD_COLOR |
                                      cv2.IMREAD_IGNORE_ORIENTATION)
        records['image'] = cv2.cvtColor(records['image'], cv2.COLOR_BGR2RGB)
        records['score'] = records['score'] if 'score' in records else 1
        records = self.transform(records)
        # print('records', records)
        return records


@register
@serializable
class KeypointTopDownCocoDataset(KeypointTopDownBaseDataset):
    """COCO dataset for top-down pose estimation. Adapted from
        https://github.com/leoxiaobin/deep-high-resolution-net.pytorch
        Copyright (c) Microsoft, under the MIT License.

    The dataset loads raw features and apply specified transforms
    to return a dict containing the image tensors and other information.

    COCO keypoint indexes:

        0: 'nose',
        1: 'left_eye',
        2: 'right_eye',
        3: 'left_ear',
        4: 'right_ear',
        5: 'left_shoulder',
        6: 'right_shoulder',
        7: 'left_elbow',
        8: 'right_elbow',
        9: 'left_wrist',
        10: 'right_wrist',
        11: 'left_hip',
        12: 'right_hip',
        13: 'left_knee',
        14: 'right_knee',
        15: 'left_ankle',
        16: 'right_ankle'

    Args:
        dataset_dir (str): Root path to the dataset.
        image_dir (str): Path to a directory where images are held.
        anno_path (str): Relative path to the annotation file.
        num_joints (int): Keypoint numbers
        trainsize (list):[w, h] Image target size
        transform (composed(operators)): A sequence of data transforms.
        bbox_file (str): Path to a detection bbox file
            Default: None.
        use_gt_bbox (bool): Whether to use ground truth bbox
            Default: True.
        pixel_std (int): The pixel std of the scale
            Default: 200.
        image_thre (float): The threshold to filter the detection box
            Default: 0.0.
    """

    def __init__(self,
                 dataset_dir,
                 image_dir,
                 anno_path,
                 num_joints,
                 trainsize,
                 transform=[],
                 bbox_file=None,
                 use_gt_bbox=True,
                 pixel_std=200,
                 image_thre=0.0):
        super().__init__(dataset_dir, image_dir, anno_path, num_joints,
                         transform)

        self.bbox_file = bbox_file
        self.use_gt_bbox = use_gt_bbox
        self.trainsize = trainsize
        self.pixel_std = pixel_std
        self.image_thre = image_thre
        self.dataset_name = 'coco'

    def parse_dataset(self):
        if self.use_gt_bbox:
            self.db = self._load_coco_keypoint_annotations()
        else:
            self.db = self._load_coco_person_detection_results()

    def _load_coco_keypoint_annotations(self):
        coco = COCO(self.get_anno())
        img_ids = coco.getImgIds()
        gt_db = []
        for index in img_ids:
            im_ann = coco.loadImgs(index)[0]
            width = im_ann['width']
            height = im_ann['height']
            file_name = im_ann['file_name']
            im_id = int(im_ann["id"])

            annIds = coco.getAnnIds(imgIds=index, iscrowd=False)
            objs = coco.loadAnns(annIds)

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
                if max(obj['keypoints']) == 0:
                    continue

                joints = np.zeros(
                    (self.ann_info['num_joints'], 3), dtype=np.float)
                joints_vis = np.zeros(
                    (self.ann_info['num_joints'], 3), dtype=np.float)
                for ipt in range(self.ann_info['num_joints']):
                    joints[ipt, 0] = obj['keypoints'][ipt * 3 + 0]
                    joints[ipt, 1] = obj['keypoints'][ipt * 3 + 1]
                    joints[ipt, 2] = 0
                    t_vis = obj['keypoints'][ipt * 3 + 2]
                    if t_vis > 1:
                        t_vis = 1
                    joints_vis[ipt, 0] = t_vis
                    joints_vis[ipt, 1] = t_vis
                    joints_vis[ipt, 2] = 0

                center, scale = self._box2cs(obj['clean_bbox'][:4])
                rec.append({
                    'image_file': os.path.join(self.img_prefix, file_name),
                    'center': center,
                    'scale': scale,
                    'joints': joints,
                    'joints_vis': joints_vis,
                    'im_id': im_id,
                })
            gt_db.extend(rec)

        return gt_db

    def _box2cs(self, box):
        x, y, w, h = box[:4]
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5
        aspect_ratio = self.trainsize[0] * 1.0 / self.trainsize[1]

        if w > aspect_ratio * h:
            h = w * 1.0 / aspect_ratio
        elif w < aspect_ratio * h:
            w = h * aspect_ratio
        scale = np.array(
            [w * 1.0 / self.pixel_std, h * 1.0 / self.pixel_std],
            dtype=np.float32)
        if center[0] != -1:
            scale = scale * 1.25

        return center, scale

    def _load_coco_person_detection_results(self):
        all_boxes = None
        bbox_file_path = os.path.join(self.dataset_dir, self.bbox_file)
        with open(bbox_file_path, 'r') as f:
            all_boxes = json.load(f)

        if not all_boxes:
            print('=> Load %s fail!' % bbox_file_path)
            return None

        kpt_db = []
        for n_img in range(0, len(all_boxes)):
            det_res = all_boxes[n_img]
            if det_res['category_id'] != 1:
                continue
            file_name = det_res[
                'filename'] if 'filename' in det_res else '%012d.jpg' % det_res[
                    'image_id']
            img_name = os.path.join(self.img_prefix, file_name)
            box = det_res['bbox']
            score = det_res['score']
            im_id = int(det_res['image_id'])

            if score < self.image_thre:
                continue

            center, scale = self._box2cs(box)
            joints = np.zeros((self.ann_info['num_joints'], 3), dtype=np.float)
            joints_vis = np.ones(
                (self.ann_info['num_joints'], 3), dtype=np.float)
            kpt_db.append({
                'image_file': img_name,
                'im_id': im_id,
                'center': center,
                'scale': scale,
                'score': score,
                'joints': joints,
                'joints_vis': joints_vis,
            })

        return kpt_db
