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

from paddle.utils.image_util import *
import random
from PIL import Image
from PIL import ImageDraw
import numpy as np
import xml.etree.ElementTree
import os
import time
import copy
import six
from collections import deque

from roidbs import JsonDataset
import data_utils
from config import cfg
import segm_utils


def roidb_reader(roidb, mode):
    im, im_scales = data_utils.get_image_blob(roidb, mode)
    im_id = roidb['id']
    im_height = np.round(roidb['height'] * im_scales)
    im_width = np.round(roidb['width'] * im_scales)
    im_info = np.array([im_height, im_width, im_scales], dtype=np.float32)
    if mode == 'test' or mode == 'infer':
        return im, im_info, im_id

    gt_boxes = roidb['gt_boxes'].astype('float32')
    gt_classes = roidb['gt_classes'].astype('int32')
    is_crowd = roidb['is_crowd'].astype('int32')
    segms = roidb['segms']

    outs = (im, gt_boxes, gt_classes, is_crowd, im_info, im_id)

    if cfg.MASK_ON:
        gt_masks = []
        valid = True
        segms = roidb['segms']
        assert len(segms) == is_crowd.shape[0]
        for i in range(len(roidb['segms'])):
            segm, iscrowd = segms[i], is_crowd[i]
            gt_segm = []
            if iscrowd:
                gt_segm.append([[0, 0]])
            else:
                for poly in segm:
                    if len(poly) == 0:
                        valid = False
                        break
                    gt_segm.append(np.array(poly).reshape(-1, 2))
            if (not valid) or len(gt_segm) == 0:
                break
            gt_masks.append(gt_segm)
        outs = outs + (gt_masks, )
    return outs


def coco(mode,
         batch_size=None,
         total_batch_size=None,
         padding_total=False,
         shuffle=False):
    if 'coco2014' in cfg.dataset:
        cfg.train_file_list = 'annotations/instances_train2014.json'
        cfg.train_data_dir = 'train2014'
        cfg.val_file_list = 'annotations/instances_val2014.json'
        cfg.val_data_dir = 'val2014'
    elif 'coco2017' in cfg.dataset:
        cfg.train_file_list = 'annotations/instances_train2017.json'
        cfg.train_data_dir = 'train2017'
        cfg.val_file_list = 'annotations/instances_val2017.json'
        cfg.val_data_dir = 'val2017'
    else:
        raise NotImplementedError('Dataset {} not supported'.format(
            cfg.dataset))
    cfg.mean_value = np.array(cfg.pixel_means)[np.newaxis,
                                               np.newaxis, :].astype('float32')
    total_batch_size = total_batch_size if total_batch_size else batch_size
    if mode != 'infer':
        assert total_batch_size % batch_size == 0
    if mode == 'train':
        cfg.train_file_list = os.path.join(cfg.data_dir, cfg.train_file_list)
        cfg.train_data_dir = os.path.join(cfg.data_dir, cfg.train_data_dir)
    elif mode == 'test' or mode == 'infer':
        cfg.val_file_list = os.path.join(cfg.data_dir, cfg.val_file_list)
        cfg.val_data_dir = os.path.join(cfg.data_dir, cfg.val_data_dir)
    json_dataset = JsonDataset(train=(mode == 'train'))
    roidbs = json_dataset.get_roidb()

    print("{} on {} with {} roidbs".format(mode, cfg.dataset, len(roidbs)))

    def padding_minibatch(batch_data):
        if len(batch_data) == 1:
            return batch_data

        max_shape = np.array([data[0].shape for data in batch_data]).max(axis=0)

        padding_batch = []
        for data in batch_data:
            im_c, im_h, im_w = data[0].shape[:]
            padding_im = np.zeros(
                (im_c, max_shape[1], max_shape[2]), dtype=np.float32)
            padding_im[:, :im_h, :im_w] = data[0]
            padding_batch.append((padding_im, ) + data[1:])
        return padding_batch

    def reader():
        if mode == "train":
            if shuffle:
                roidb_perm = deque(np.random.permutation(roidbs))
            else:
                roidb_perm = deque(roidbs)
            roidb_cur = 0
            count = 0
            batch_out = []
            device_num = total_batch_size / batch_size
            while True:
                roidb = roidb_perm[0]
                roidb_cur += 1
                roidb_perm.rotate(-1)
                if roidb_cur >= len(roidbs):
                    if shuffle:
                        roidb_perm = deque(np.random.permutation(roidbs))
                    else:
                        roidb_perm = deque(roidbs)
                    roidb_cur = 0
                # im, gt_boxes, gt_classes, is_crowd, im_info, im_id, gt_masks
                datas = roidb_reader(roidb, mode)
                if datas[1].shape[0] == 0:
                    continue
                if cfg.MASK_ON:
                    if len(datas[-1]) != datas[1].shape[0]:
                        continue
                batch_out.append(datas)
                if not padding_total:
                    if len(batch_out) == batch_size:
                        yield padding_minibatch(batch_out)
                        count += 1
                        batch_out = []
                else:
                    if len(batch_out) == total_batch_size:
                        batch_out = padding_minibatch(batch_out)
                        for i in range(device_num):
                            sub_batch_out = []
                            for j in range(batch_size):
                                sub_batch_out.append(batch_out[i * batch_size +
                                                               j])
                            yield sub_batch_out
                            count += 1
                            sub_batch_out = []
                        batch_out = []
                iter_id = count // device_num
                if iter_id >= cfg.max_iter:
                    return
        elif mode == "test":
            batch_out = []
            for roidb in roidbs:
                im, im_info, im_id = roidb_reader(roidb, mode)
                batch_out.append((im, im_info, im_id))
                if len(batch_out) == batch_size:
                    yield batch_out
                    batch_out = []
            if len(batch_out) != 0:
                yield batch_out

        else:
            for roidb in roidbs:
                if cfg.image_name not in roidb['image']:
                    continue
                im, im_info, im_id = roidb_reader(roidb, mode)
                batch_out = [(im, im_info, im_id)]
                yield batch_out

    return reader


def train(batch_size, total_batch_size=None, padding_total=False, shuffle=True):
    return coco(
        'train', batch_size, total_batch_size, padding_total, shuffle=shuffle)


def test(batch_size, total_batch_size=None, padding_total=False):
    return coco('test', batch_size, total_batch_size, shuffle=False)


def infer():
    return coco('infer')
