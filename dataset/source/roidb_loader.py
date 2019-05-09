# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

# function:
#   load coco data from local files

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os
import numpy as np

def load_from_json(json_path, samples=-1):
    """ load roidb from json file
    
    
    
    
    Args:
        @json_path (str): json file path
        @samples (int): samples to load, -1 means all

    Returns:
        list of loaded sample(but no binary image)
    """
    import matplotlib
    matplotlib.use('Agg')
    from pycocotools.coco import COCO

    assert json_path.endswith('.json'), 'invalid json file[%s]' % (json_path)

    dataset = COCO(json_path)
    img_ids = dataset.getImgIds()
    roidb = []
    ct = 0
    for img_id in img_ids:
        img_anno = dataset.loadImgs(img_id)[0]
        im_filename = img_anno['file_name']
        im_w = img_anno['width']
        im_h = img_anno['height']

        ins_anno_ids = dataset.getAnnIds(imgIds=img_id, iscrowd=False)
        trainid_to_datasetid = dict({i + 1: cid for i, cid in enumerate(dataset.getCatIds())}) # 0 for bg
        datasetid_to_trainid = dict({cid: tid for tid, cid in trainid_to_datasetid.items()})
        instances = dataset.loadAnns(ins_anno_ids)

        # sanitize bboxes
        valid_instances = []
        for inst in instances:
            x, y, box_w, box_h = inst['bbox']
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(im_w - 1, x1 + max(0, box_w - 1))
            y2 = min(im_h - 1, y1 + max(0, box_h - 1))
            if inst['area'] > 0 and x2 >= x1 and y2 >= y1:
                inst['clean_bbox'] = [x1, y1, x2, y2]
                valid_instances.append(inst)
        num_instance = len(valid_instances)

        gt_bbox = np.zeros((num_instance, 4), dtype=np.float32)
        gt_class = np.zeros((num_instance, ), dtype=np.int32)
        gt_poly = [None] * num_instance

        for i, inst in enumerate(valid_instances):
            cls = datasetid_to_trainid[inst['category_id']]
            gt_bbox[i, :] = inst['clean_bbox']
            gt_class[i] = cls
            gt_poly[i] = inst['segmentation']

        roi_rec = {
            'image_url': im_filename,
            'im_id': img_id,
            'h': im_h,
            'w': im_w,
            'gt_class': gt_class,
            'gt_bbox': gt_bbox,
            'gt_poly': gt_poly,
            'flipped': False}

        roidb.append(roi_rec)
        ct += 1
        if samples > 0 and ct >= samples:
            break

    assert type(roidb) is list, 'invalid data type from roidb'
    return roidb


def load_from_pickle(pickle_path, samples=-1):
    """ load roidb data from pickled file
    """
    import pickle as pkl

    assert pickle_path.endswith('.roidb'), 'invalid roidb file[%s]' % (pickle_path)
    with open(pickle_path, 'rb') as f:
        roidb = f.read()
        roidb = pkl.loads(roidb)
        assert type(roidb) is list, 'invalid data type from roidb'
    if samples > 0 and samples < len(roidb):
        roidb = roidb[:samples]

    return roidb


def load(fnames, samples=-1):
    """ load coco data from list of files in 'fnames',
    Args:
        @fnames (list of str): file names for data, eg:
            instances_val2017.json or COCO17_val2017.roidb
        @samples (int): number of samples to load, default to all

    Returns:
        list of loaded samples(but no binary image)
    """
    fnames = [fnames] if type(fnames) is str else fnames
    roidb = []
    for fn in fnames:
        if fn.endswith('.json'):
            roidb += load_from_json(fn, samples)
        elif fn.endswith('.roidb'):
            roidb += load_from_pickle(fn, samples)
        else:
            raise ValueError('invalid file type when load roidb data from file[%s]' % (fn))

    return roidb
