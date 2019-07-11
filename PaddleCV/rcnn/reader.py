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

import random
import numpy as np
import xml.etree.ElementTree
import os
import time
import copy
import six
import cv2
from collections import deque

from roidbs import JsonDataset
import data_utils
from config import cfg
import segm_utils
num_trainers = int(os.environ.get('PADDLE_TRAINERS_NUM', 1))


def roidb_reader(roidb, mode):
    im, im_scales = data_utils.get_image_blob(roidb, mode)
    im_id = roidb['id']
    im_height = np.round(roidb['height'] * im_scales)
    im_width = np.round(roidb['width'] * im_scales)
    im_info = np.array([im_height, im_width, im_scales], dtype=np.float32)
    if mode == 'val':
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
         shuffle=False,
         shuffle_seed=None):
    total_batch_size = total_batch_size if total_batch_size else batch_size
    assert total_batch_size % batch_size == 0
    json_dataset = JsonDataset(mode)
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
                if shuffle_seed is not None:
                    np.random.seed(shuffle_seed)
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
                if iter_id >= cfg.max_iter * num_trainers:
                    return
        elif mode == "val":
            batch_out = []
            for roidb in roidbs:
                im, im_info, im_id = roidb_reader(roidb, mode)
                batch_out.append((im, im_info, im_id))
                if len(batch_out) == batch_size:
                    yield batch_out
                    batch_out = []
            if len(batch_out) != 0:
                yield batch_out

    return reader


def train(batch_size,
          total_batch_size=None,
          padding_total=False,
          shuffle=True,
          shuffle_seed=None):
    return coco(
        'train',
        batch_size,
        total_batch_size,
        padding_total,
        shuffle=shuffle,
        shuffle_seed=shuffle_seed)


def test(batch_size, total_batch_size=None, padding_total=False):
    return coco('val', batch_size, total_batch_size, shuffle=False)


def infer(file_path):
    def reader():
        if not os.path.exists(file_path):
            raise ValueError("Image path [%s] does not exist." % (file_path))
        im = cv2.imread(file_path)
        im = im.astype(np.float32, copy=False)
        im -= cfg.pixel_means
        im_height, im_width, channel = im.shape
        channel_swap = (2, 0, 1)  #(channel, height, width)
        im = im.transpose(channel_swap)
        im_info = np.array([im_height, im_width, 1.0], dtype=np.float32)
        yield [(im, im_info)]

    return reader
