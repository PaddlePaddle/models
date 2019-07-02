# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved
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
#
# Based on:
# --------------------------------------------------------
# Detectron
# Copyright (c) 2017-present, Facebook, Inc.
# Licensed under the Apache License, Version 2.0;
# Written by Ross Girshick
# --------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import copy
import logging
import numpy as np
import os
import scipy.sparse
import random
import time
import matplotlib
matplotlib.use('Agg')
from pycocotools.coco import COCO
import box_utils
import segm_utils
from config import cfg
from data_utils import DatasetPath

logger = logging.getLogger(__name__)


class JsonDataset(object):
    """A class representing a COCO json dataset."""

    def __init__(self, mode):
        print('Creating: {}'.format(cfg.dataset))
        self.name = cfg.dataset
        self.is_train = mode == 'train'
        data_path = DatasetPath(mode)
        data_dir = data_path.get_data_dir()
        file_list = data_path.get_file_list()
        self.image_directory = data_dir
        self.COCO = COCO(file_list)
        # Set up dataset classes
        category_ids = self.COCO.getCatIds()
        categories = [c['name'] for c in self.COCO.loadCats(category_ids)]
        self.category_to_id_map = dict(zip(categories, category_ids))
        self.classes = ['__background__'] + categories
        self.num_classes = len(self.classes)
        self.json_category_id_to_contiguous_id = {
            v: i + 1
            for i, v in enumerate(self.COCO.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k
            for k, v in self.json_category_id_to_contiguous_id.items()
        }

    def get_roidb(self):
        """Return an roidb corresponding to the json dataset. Optionally:
           - include ground truth boxes in the roidb
           - add proposals specified in a proposals file
           - filter proposals based on a minimum side length
           - filter proposals that intersect with crowd regions
        """
        image_ids = self.COCO.getImgIds()
        image_ids.sort()
        roidb = copy.deepcopy(self.COCO.loadImgs(image_ids))
        for entry in roidb:
            self._prep_roidb_entry(entry)
        if self.is_train:
            # Include ground-truth object annotations
            start_time = time.time()
            for entry in roidb:
                self._add_gt_annotations(entry)
            end_time = time.time()
            print('_add_gt_annotations took {:.3f}s'.format(end_time -
                                                            start_time))
            if cfg.TRAIN.use_flipped:
                print('Appending horizontally-flipped training examples...')
                self._extend_with_flipped_entries(roidb)
        print('Loaded dataset: {:s}'.format(self.name))
        print('{:d} roidb entries'.format(len(roidb)))
        if self.is_train:
            self._filter_for_training(roidb)
        return roidb

    def _prep_roidb_entry(self, entry):
        """Adds empty metadata fields to an roidb entry."""
        # Make file_name an abs path
        im_path = os.path.join(self.image_directory, entry['file_name'])
        #assert os.path.exists(im_path), 'Image \'{}\' not found'.format(im_path)
        entry['image'] = im_path
        entry['flipped'] = False
        # Empty placeholders
        entry['gt_boxes'] = np.empty((0, 4), dtype=np.float32)
        entry['gt_classes'] = np.empty((0), dtype=np.int32)
        entry['gt_id'] = np.empty((0), dtype=np.int32)
        entry['is_crowd'] = np.empty((0), dtype=np.bool)
        entry['segms'] = []
        # Remove unwanted fields that come from the json file (if they exist)
        for k in ['date_captured', 'url', 'license', 'file_name']:
            if k in entry:
                del entry[k]

    def _add_gt_annotations(self, entry):
        """Add ground truth annotation metadata to an roidb entry."""
        count = 0
        #for k in self.category_to_id_map:
        #    imgs = self.COCO.getImgIds(catIds=(self.category_to_id_map[k]))
        #    count += len(imgs)
        ann_ids = self.COCO.getAnnIds(imgIds=entry['id'], iscrowd=None)
        objs = self.COCO.loadAnns(ann_ids)
        # Sanitize bboxes -- some are invalid
        valid_objs = []
        valid_segms = []
        width = entry['width']
        height = entry['height']
        for obj in objs:
            if isinstance(obj['segmentation'], list):
                # Valid polygons have >= 3 points, so require >= 6 coordinates
                obj['segmentation'] = [
                    p for p in obj['segmentation'] if len(p) >= 6
                ]
            if obj['area'] < cfg.TRAIN.gt_min_area:
                continue
            if 'ignore' in obj and obj['ignore'] == 1:
                continue
            # Convert form (x1, y1, w, h) to (x1, y1, x2, y2)
            x1, y1, x2, y2 = box_utils.xywh_to_xyxy(obj['bbox'])
            x1, y1, x2, y2 = box_utils.clip_xyxy_to_image(x1, y1, x2, y2,
                                                          height, width)
            # Require non-zero seg area and more than 1x1 box size
            if obj['area'] > 0 and x2 > x1 and y2 > y1:
                obj['clean_bbox'] = [x1, y1, x2, y2]
                valid_objs.append(obj)
                valid_segms.append(obj['segmentation'])

        num_valid_objs = len(valid_objs)

        gt_boxes = np.zeros((num_valid_objs, 4), dtype=entry['gt_boxes'].dtype)
        gt_id = np.zeros((num_valid_objs), dtype=np.int64)
        gt_classes = np.zeros((num_valid_objs), dtype=entry['gt_classes'].dtype)
        is_crowd = np.zeros((num_valid_objs), dtype=entry['is_crowd'].dtype)
        for ix, obj in enumerate(valid_objs):
            cls = self.json_category_id_to_contiguous_id[obj['category_id']]
            gt_boxes[ix, :] = obj['clean_bbox']
            gt_classes[ix] = cls
            gt_id[ix] = np.int64(obj['id'])
            is_crowd[ix] = obj['iscrowd']

        entry['gt_boxes'] = np.append(entry['gt_boxes'], gt_boxes, axis=0)
        entry['gt_classes'] = np.append(entry['gt_classes'], gt_classes)
        entry['gt_id'] = np.append(entry['gt_id'], gt_id)
        entry['is_crowd'] = np.append(entry['is_crowd'], is_crowd)
        entry['segms'].extend(valid_segms)

    def _extend_with_flipped_entries(self, roidb):
        """Flip each entry in the given roidb and return a new roidb that is the
        concatenation of the original roidb and the flipped entries.
        "Flipping" an entry means that that image and associated metadata (e.g.,
        ground truth boxes and object proposals) are horizontally flipped.
        """
        flipped_roidb = []
        for entry in roidb:
            width = entry['width']
            gt_boxes = entry['gt_boxes'].copy()
            oldx1 = gt_boxes[:, 0].copy()
            oldx2 = gt_boxes[:, 2].copy()
            gt_boxes[:, 0] = width - oldx2 - 1
            gt_boxes[:, 2] = width - oldx1 - 1
            assert (gt_boxes[:, 2] >= gt_boxes[:, 0]).all()
            flipped_entry = {}
            dont_copy = ('gt_boxes', 'flipped', 'segms')
            for k, v in entry.items():
                if k not in dont_copy:
                    flipped_entry[k] = v
            flipped_entry['gt_boxes'] = gt_boxes
            flipped_entry['segms'] = segm_utils.flip_segms(
                entry['segms'], entry['height'], entry['width'])
            flipped_entry['flipped'] = True
            flipped_roidb.append(flipped_entry)
        roidb.extend(flipped_roidb)

    def _filter_for_training(self, roidb):
        """Remove roidb entries that have no usable RoIs based on config settings.
        """

        def is_valid(entry):
            # Valid images have:
            #   (1) At least one groundtruth RoI OR
            #   (2) At least one background RoI
            gt_boxes = entry['gt_boxes']
            # image is only valid if such boxes exist
            valid = len(gt_boxes) > 0
            return valid

        num = len(roidb)
        filtered_roidb = [entry for entry in roidb if is_valid(entry)]
        num_after = len(filtered_roidb)
        print('Filtered {} roidb entries: {} -> {}'.format(num - num_after, num,
                                                           num_after))
