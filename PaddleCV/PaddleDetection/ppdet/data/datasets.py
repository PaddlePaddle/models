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

from __future__ import division

import copy
import os

import numpy as np
import cv2
from pycocotools.coco import COCO


__all__ = ['COCODataSet', 'PascalVocDataSet', 'ImageFolder', 'DataSet',
           'ListDataSet', 'StreamDataSet']


class DataSet(object):
    def __init__(self, ):
        super(DataSet, self).__init__()

    @property
    def mode(self):
        if hasattr(self, '_mode'):
            return self._mode
        if hasattr(self, '__getitem__') and hasattr(self, '__len__'):
            self._mode = 'indexable'
        elif hasattr(self, 'next') or hasattr(self, '__next__'):
            self._mode = 'iterable'
        else:
            raise "dataset should be either indexable or iterable"
        return self._mode

    def _read_image(self, path):
        with open(path, 'rb') as f:
            data = np.frombuffer(f.read(), dtype='uint8')
            img = cv2.imdecode(data, 1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img


class COCODataSet(DataSet):
    def __init__(self,
                 root_dir,
                 annotation_file,
                 image_dir,
                 use_mask=False,
                 use_crowd=False,
                 use_empty=False):
        super(COCODataSet, self).__init__()
        assert annotation_file is not None and image_dir is not None
        self._annotation_file = annotation_file
        self.root_dir = root_dir
        self.image_dir = image_dir
        self.use_mask = use_mask
        self.use_crowd = use_crowd
        self.use_empty = use_empty

    @property
    def annotation_file(self):
        return os.path.join(self.root_dir, self._annotation_file)

    @property
    def aspect_ratios(self):
        if not hasattr(self, '_aspect_ratios'):
            if not hasattr(self, 'samples'):
                self._load()
            self._aspect_ratios = [s['width'] / s['height']
                                   for s in self.samples]
        return self._aspect_ratios

    def __len__(self):
        if not hasattr(self, 'samples'):
            self._load()
        return len(self.samples)

    def __getitem__(self, idx):
        if not hasattr(self, 'samples'):
            self._load()
        sample = copy.deepcopy(self.samples[idx])
        sample['image'] = self._read_image(sample['file'])
        return sample

    def _load(self):
        coco = COCO(self.annotation_file)
        img_ids = coco.getImgIds()
        imgs = coco.loadImgs(img_ids)
        class_map = {v: i + 1 for i, v in enumerate(coco.getCatIds())}
        samples = []

        for img in imgs:
            sample = {}
            img_path = os.path.join(
                self.root_dir, self.image_dir, img['file_name'])
            sample['file'] = img_path
            sample['id'] = img['id']
            width = img['width']
            height = img['height']
            sample['width'] = width
            sample['height'] = height
            sample['orig_width'] = width
            sample['orig_height'] = height
            sample['scale'] = 1.
            ann_ids = coco.getAnnIds(imgIds=img['id'], iscrowd=self.use_crowd)
            anns = coco.loadAnns(ann_ids)

            gt_box = []
            gt_label = []
            is_crowd = []
            if self.use_mask:
                gt_poly = []

            for ann in anns:
                x1, y1, w, h = ann['bbox']
                x2 = x1 + w - 1
                y2 = y1 + h - 1

                x1 = np.clip(x1, 0, width - 1)
                x2 = np.clip(x2, 0, width - 1)
                y1 = np.clip(y1, 0, height - 1)
                y2 = np.clip(y2, 0, height - 1)

                if ann['area'] <= 0 or x2 < x1 or y2 < y1:
                    continue
                if self.use_mask:
                    if ann['iscrowd']:
                        poly = [np.zeros((1, 2), dtype=np.float32)]
                    else:
                        poly = [np.array(p, dtype=np.float32).reshape(-1, 2)
                                for p in ann['segmentation'] if len(p) > 6]
                    if not poly:
                        continue
                    gt_poly.append(poly)

                gt_label.append(ann['category_id'])
                gt_box.append([x1, y1, x2, y2])
                is_crowd.append(int(ann['iscrowd']))

            gt_box = np.array(gt_box, dtype=np.float32)
            gt_label = np.array([class_map[cls] for cls in gt_label],
                                dtype=np.int32)[:, np.newaxis]

            sample['gt_box'] = gt_box
            sample['gt_label'] = gt_label
            sample['gt_score'] = np.ones_like(gt_label, dtype=np.float32)
            sample['is_crowd'] = np.array(is_crowd, np.int32)[:, np.newaxis]
            if self.use_mask:
                sample['gt_poly'] = gt_poly

            if gt_label.size == 0 and not self.use_empty:
                continue
            samples.append(sample)

        self.samples = samples


class PascalVocDataSet(DataSet):
    def __init__(self,
                 root_dir='VOCdevkit/VOC2012',
                 subset='train',
                 use_empty=False):
        super(PascalVocDataSet, self).__init__()
        self.root_dir = root_dir
        self.subset = subset
        self.use_empty = use_empty
        label_map = {
            'aeroplane': 1,
            'bicycle': 2,
            'bird': 3,
            'boat': 4,
            'bottle': 5,
            'bus': 6,
            'car': 7,
            'cat': 8,
            'chair': 9,
            'cow': 10,
            'diningtable': 11,
            'dog': 12,
            'horse': 13,
            'motorbike': 14,
            'person': 15,
            'pottedplant': 16,
            'sheep': 17,
            'sofa': 18,
            'train': 19,
            'tvmonitor': 20
        }
        self.label_map = label_map

    @property
    def annotation_file(self):
        return os.path.join(
            self.root_dir, 'ImageSets/Main/{}.txt'.format(self.subset))

    @property
    def image_dir(self):
        return os.path.join(self.root_dir, 'JPEGImages')

    @property
    def anno_dir(self):
        return os.path.join(self.root_dir, 'Annotations')

    @property
    def aspect_ratios(self):
        if not hasattr(self, '_aspect_ratios'):
            if not hasattr(self, 'samples'):
                self._load()
            self._aspect_ratios = [s['width'] / s['height']
                                   for s in self.samples]
        return self._aspect_ratios

    def __len__(self):
        if not hasattr(self, 'samples'):
            self._load()
        return len(self.samples)

    def __getitem__(self, idx):
        if not hasattr(self, 'samples'):
            self._load()
        sample = copy.deepcopy(self.samples[idx])
        sample['image'] = self._read_image(sample['file'])
        return sample

    def _load(self):
        import xml.etree.ElementTree as ET
        indices = open(self.annotation_file).readlines()

        def _get(root, path):
            nodes = path.split('.')
            for node in nodes:
                root = root.find(node)
            return root.text

        def parse_xml(idx):
            idx = idx.strip()
            sample = {'id': idx}
            xml_file = os.path.join(self.anno_dir, idx + '.xml')
            tree = ET.parse(xml_file)

            objs = tree.findall('object')
            h = int(_get(tree, 'size.height'))
            w = int(_get(tree, 'size.width'))
            sample['file'] = os.path.join(self.image_dir,
                                          _get(tree, 'filename'))
            sample['width'] = w
            sample['height'] = h
            sample['orig_width'] = w
            sample['orig_height'] = h
            gt_box = []
            gt_label = []
            difficult = []
            for obj in objs:
                gt_label.append(self.label_map[_get(obj, 'name')])
                difficult.append(int(_get(obj, 'difficult')))
                x1 = max(0, float(_get(obj, 'bndbox.xmin')))
                y1 = max(0, float(_get(obj, 'bndbox.ymin')))
                x2 = min(w - 1, float(_get(obj, 'bndbox.xmax')))
                y2 = min(h - 1, float(_get(obj, 'bndbox.ymax')))
                gt_box.append([x1, y1, x2, y2])
            if not self.use_empty and len(gt_label) == 0:
                return None
            sample['gt_box'] = np.array(gt_box, dtype=np.float32)
            sample['gt_label'] = np.array(
                gt_label, dtype=np.int32)[:, np.newaxis]
            sample['gt_score'] = np.ones([len(gt_label), 1], dtype=np.float32)
            sample['is_crowd'] = np.zeros([len(gt_label), 1], dtype=np.int32)
            sample['gt_poly'] = []
            sample['difficult'] = np.array(
                difficult, dtype=np.int32)[:, np.newaxis]
            return sample

        samples = [parse_xml(idx) for idx in indices]
        self.samples = list(filter(lambda x: x is not None, samples))


class ListDataSet(DataSet):
    def __init__(self, list_fn, *args, **kwargs):
        super(ListDataSet, self).__init__()
        self.samples = list_fn(*args, **kwargs)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = copy.deepcopy(self.samples[idx])
        if 'image' not in sample and 'file' in sample:
            img = self._read_image(sample['file'])
            h, w, _ = img.shape
            sample['image'] = img
            if 'width' not in sample:
                sample['width'] = w
            if 'height' not in sample:
                sample['height'] = h
            sample['orig_width'] = sample['width']
            sample['orig_height'] = sample['height']
        return sample


class ImageFolder(ListDataSet):
    def __init__(self, root_dir=None):
        def find_image_files(dir):
            image_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.ppm', '.pgm']

            samples = []
            for target in sorted(os.listdir(dir)):
                path = os.path.join(dir, target)
                if not os.path.isfile(path):
                    continue
                if os.path.splitext(target)[1].lower() in image_exts:
                    samples.append({'file': path})

            return samples
        super(ImageFolder, self).__init__(find_image_files)


class StreamDataSet(DataSet):
    def __init__(self, generator):
        super(StreamDataSet, self).__init__()
        self.generator = generator

    def __next__(self):
        return next(self.generator)

    # python 2 compatibility
    next = __next__
